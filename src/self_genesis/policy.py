"""Trainable recurrent policy, with state owned by individual agents."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from self_genesis.config import CAPACITY_PRESETS
from self_genesis.batched_policy import BatchedObservation, BatchedPolicyState
from self_genesis.entity_memory import BatchedEntityMemory
from self_genesis.encounter import EncounterExperience, Observation
from self_genesis.world import Action


@dataclass(frozen=True)
class EntityMemoryEntry:
    appearance: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class PolicyState:
    memory: torch.Tensor
    affect: torch.Tensor
    entities: tuple[EntityMemoryEntry, ...] = ()


class RecurrentPolicy(nn.Module):
    """Shared weights only; no individual state or identity labels.

    Callback-local Thought starts at zero and recurs with fixed context through
    one shared Linear/ReLU core for think_steps (>=16). Shallow mode retains the
    original single Linear/tanh comparison. Previous memory and unlabeled affect
    feed Thought and the single post-loop memory update. New affect depends on
    the observation, Thought and updated memory, and feeds the following callback.
    Message slots have no assigned meaning.
    """

    def __init__(self, appearance_dim: int, *, vocabulary_size: int = 4,
                 max_message_length: int = 3, memory_dim: int = 16,
                 affect_dim: int = 4, entity_memory_dim: int = 16,
                 think_steps: int = 16, thought_mode: str = "recurrent"):
        super().__init__()
        for name, value, minimum in (
            ("appearance_dim", appearance_dim, 1),
            ("vocabulary_size", vocabulary_size, 1),
            ("max_message_length", max_message_length, 0),
            ("memory_dim", memory_dim, 1), ("affect_dim", affect_dim, 1),
            ("entity_memory_dim", entity_memory_dim, 0),
            ("think_steps", think_steps, 16),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if thought_mode not in ("recurrent", "shallow"):
            raise ValueError("thought_mode must be recurrent or shallow")
        self.think_steps = think_steps
        self.thought_mode = thought_mode
        self.appearance_dim = appearance_dim
        self.vocabulary_size = vocabulary_size
        self.max_message_length = max_message_length
        self.memory_dim = memory_dim
        self.affect_dim = affect_dim
        self.entity_memory_dim = entity_memory_dim
        # Four resources, role, three partner-action categories, callback phase,
        # appearance, and ordered message slots (including an empty category).
        input_dim = 9 + appearance_dim + max_message_length * (vocabulary_size + 1)
        context_dim = input_dim + memory_dim + affect_dim + entity_memory_dim
        self.thought = nn.Linear(
            context_dim + (memory_dim if thought_mode == "recurrent" else 0), memory_dim)
        self.memory_update = nn.GRUCell(memory_dim + affect_dim, memory_dim)
        self.affect_update = nn.Linear(input_dim + 2 * memory_dim, affect_dim)
        self.message_head = nn.Linear(memory_dim, vocabulary_size)
        self.action_head = nn.Linear(memory_dim, 2)
        self.value_head = nn.Linear(memory_dim, 1)
        if entity_memory_dim:
            self.entity_update = nn.GRUCell(
                input_dim + 2 * memory_dim + affect_dim, entity_memory_dim)
            self.encounter_update = nn.GRUCell(
                input_dim + memory_dim + affect_dim + 4, entity_memory_dim)

    @classmethod
    def from_preset(cls, appearance_dim: int, capacity: str, **overrides):
        """Build a named capacity; explicit dimension/channel overrides are allowed."""
        if capacity not in CAPACITY_PRESETS:
            raise ValueError("capacity must be small, medium, or large")
        dimensions = dict(zip(("memory_dim", "affect_dim", "entity_memory_dim"),
                              CAPACITY_PRESETS[capacity]))
        dimensions.update(overrides)
        return cls(appearance_dim, **dimensions)

    @property
    def parameter_count(self) -> int:
        """Exact shared parameter count, excluding per-agent episode state."""
        return sum(parameter.numel() for parameter in self.parameters())

    def _think(self, context, previous_memory):
        """Yield callback-local steps without retaining a trace."""
        if self.thought_mode == "shallow":
            yield torch.tanh(self.thought(context))
        else:
            # BF16 rounding near zero changes ReLU gates repeatedly through
            # the shared core and can distort full-episode gradients. Keep
            # this recurrence in the parameter dtype, with its graph intact.
            context = context.to(self.thought.weight)
            thought = torch.zeros_like(previous_memory).to(context)
            for _ in range(self.think_steps):
                with torch.autocast(device_type=context.device.type, enabled=False):
                    thought = F.relu(self.thought(torch.cat((context, thought), dim=-1)))
                yield thought

    def _recur(self, inputs, previous_memory, previous_affect, retrieved):
        context = torch.cat((inputs, previous_memory, previous_affect, retrieved), dim=-1)
        # Only Thought advances inside the loop; keep its full training graph.
        for thought in self._think(context, previous_memory):
            pass
        memory = self.memory_update(torch.cat((thought, previous_affect), dim=-1),
                                    previous_memory)
        affect = torch.tanh(self.affect_update(torch.cat((inputs, thought, memory), dim=-1)))
        return thought, memory, affect

    def initial_batch_state(self, worlds: int, observers: int, *, slots: int):
        parameter = next(self.parameters())
        entities = BatchedEntityMemory.empty(
            worlds, observers, slots, self.appearance_dim, self.entity_memory_dim,
            device=parameter.device, dtype=parameter.dtype)
        return BatchedPolicyState(parameter.new_zeros(worlds, observers, self.memory_dim),
                                  parameter.new_zeros(worlds, observers, self.affect_dim),
                                  entities)

    def _encode_batch(self, observation, state, active, communicating):
        parameter = next(self.parameters())
        shape = state.memory.shape[:2]
        if (state.memory.shape != (*shape, self.memory_dim)
                or state.affect.shape != (*shape, self.affect_dim)
                or len(shape) != 2):
            raise ValueError("Unexpected batched recurrent state shape")
        for tensor in (state.memory, state.affect, state.entities.values):
            if tensor.device != parameter.device or tensor.dtype != parameter.dtype:
                raise ValueError("State must match policy device and dtype")
        if (active.shape != shape or active.dtype != torch.bool
                or active.device != parameter.device):
            raise ValueError("active must be boolean [world, observer] on the policy device")
        for name, suffix, dtype in (
            ("resources", (4,), None),
            ("partner_appearance", (self.appearance_dim,), parameter.dtype),
            ("first", (), torch.bool),
            ("received_message", (self.max_message_length,), torch.int64),
            ("partner_action", (), torch.int64),
        ):
            tensor = getattr(observation, name)
            if (tensor.shape != (*shape, *suffix) or tensor.device != parameter.device
                    or (dtype is not None and tensor.dtype != dtype)):
                raise ValueError(f"Unexpected {name} shape, device or dtype")
        resources = observation.resources[active].to(parameter)
        appearance = observation.partner_appearance[active]
        message = observation.received_message[active]
        action = observation.partner_action[active]
        if (bool((~torch.isfinite(resources) | (resources < 0)).any())
                or not bool(torch.isfinite(appearance).all())
                or bool(((message < 0) | (message > self.vocabulary_size)).any())
                or bool(((action < 0) | (action > 2)).any())):
            raise ValueError("Active observation contains invalid resources, Appearance or categories")
        if bool(((message[..., :-1] == self.vocabulary_size)
                 & (message[..., 1:] != self.vocabulary_size)).any()):
            raise ValueError("Message padding must be a suffix")
        context = torch.cat((observation.first[active].unsqueeze(-1).to(parameter),
                             F.one_hot(action, 3).to(parameter),
                             parameter.new_full((action.numel(), 1), communicating)), dim=-1)
        encoded_message = F.one_hot(message, self.vocabulary_size + 1).flatten(1).to(parameter)
        return torch.cat((resources.log1p(), context, appearance, encoded_message), dim=-1)

    def forward_batch(self, observation: BatchedObservation, state: BatchedPolicyState, *,
                      communicating: bool, active: torch.Tensor):
        """One same-phase callback across worlds/observers, without per-agent forwards.

        Return logits, scalar values and new state. Inactive output rows are zero;
        their state is preserved. Callers sample only active rows (and skip message
        decisions when max_message_length is zero). Actor and critic share only
        permitted observations and the resulting private recurrent memory.
        """
        inputs = self._encode_batch(observation, state, active, communicating)
        retrieved = state.entities.retrieve(observation.partner_appearance, active=active)[active]
        thought, memory, affect = self._recur(inputs, state.memory[active],
                                             state.affect[active], retrieved)
        next_memory, next_affect = state.memory.clone(), state.affect.clone()
        # Indexed writes require matching dtypes; keep persistent state in FP32
        # even when eligible neural operations run under BF16 autocast.
        next_memory[active], next_affect[active] = memory.to(next_memory), affect.to(next_affect)
        logits = inputs.new_zeros(*active.shape, self.vocabulary_size if communicating else 2)
        values = inputs.new_zeros(active.shape)
        head = self.message_head if communicating else self.action_head
        logits[active] = head(memory).to(logits)
        # Value regression is sensitive to rounding as survival returns grow.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            values[active] = self.value_head(memory.to(values)).squeeze(-1)
        entities = state.entities
        if self.entity_memory_dim:
            value = inputs.new_zeros(*active.shape, self.entity_memory_dim)
            value[active] = self.entity_update(
                torch.cat((inputs, thought, memory, affect), dim=-1), retrieved).to(value)
            entities = entities.write(observation.partner_appearance, value, active=active)
        return logits, values, BatchedPolicyState(next_memory, next_affect, entities)

    def complete_encounter_batch(self, observation: BatchedObservation,
                                 state: BatchedPolicyState, *, action: torch.Tensor,
                                 gave: torch.Tensor, received: torch.Tensor,
                                 active: torch.Tensor):
        """Write resolved, participant-visible outcomes without advancing recurrence.

        action is int64 (0 NOTHING, 1 GIVE); gave/received are boolean. All have
        shape [world, observer]. Only the completion phase may reveal final actions.
        """
        inputs = self._encode_batch(observation, state, active, False)
        for tensor, dtype in ((action, torch.int64), (gave, torch.bool), (received, torch.bool)):
            if tensor.shape != active.shape or tensor.dtype != dtype or tensor.device != active.device:
                raise ValueError("Unexpected batched encounter outcome shape, device or dtype")
        if bool(((action[active] < 0) | (action[active] > 1)).any()):
            raise ValueError("Encounter action must be 0 NOTHING or 1 GIVE")
        if not self.entity_memory_dim:
            return state
        outcome = torch.cat((F.one_hot(action[active], 2), gave[active].unsqueeze(-1),
                             received[active].unsqueeze(-1)), dim=-1).to(inputs)
        value = inputs.new_zeros(*active.shape, self.entity_memory_dim)
        value[active] = self.encounter_update(
            torch.cat((inputs, state.memory[active], state.affect[active], outcome), dim=-1),
            state.entities.retrieve(observation.partner_appearance, active=active)[active]).to(value)
        return BatchedPolicyState(state.memory, state.affect, state.entities.write(
            observation.partner_appearance, value, active=active))

    def initial_state(self) -> PolicyState:
        parameter = next(self.parameters())
        return PolicyState(parameter.new_zeros(self.memory_dim),
                           parameter.new_zeros(self.affect_dim))

    def _encode(self, observation: Observation, communicating: bool) -> torch.Tensor:
        parameter = next(self.parameters())
        if observation.partner_appearance.shape != (self.appearance_dim,):
            raise ValueError("Unexpected partner appearance shape")
        message = observation.received_message
        if (len(message) > self.max_message_length
                or any(type(token) is not int or not 0 <= token < self.vocabulary_size
                       for token in message)):
            raise ValueError("Received message is outside the policy channel bounds")
        resources = parameter.new_tensor([
            observation.life, observation.points,
            observation.partner_life, observation.partner_points,
        ]).log1p()
        context = parameter.new_tensor([
            observation.first, observation.partner_action is None,
            observation.partner_action is Action.NOTHING,
            observation.partner_action is Action.GIVE, communicating,
        ])
        slots = torch.full((self.max_message_length,), self.vocabulary_size,
                           device=parameter.device, dtype=torch.long)
        if message:
            slots[:len(message)] = torch.tensor(message, device=parameter.device)
        encoded_message = F.one_hot(slots, self.vocabulary_size + 1).flatten().to(parameter)
        return torch.cat((resources, context,
                          observation.partner_appearance.to(parameter), encoded_message))

    def retrieve_entity(self, appearance: torch.Tensor, state: PolicyState) -> torch.Tensor:
        """Exact observed-Appearance lookup; collisions deliberately share a value."""
        if self.entity_memory_dim:
            for entry in state.entities:
                if torch.equal(entry.appearance, appearance):
                    return entry.value
        return state.memory.new_zeros(self.entity_memory_dim)

    def forward(self, observation: Observation, state: PolicyState, *,
                communicating: bool) -> tuple[torch.Tensor, PolicyState]:
        inputs = self._encode(observation, communicating)
        retrieved = self.retrieve_entity(observation.partner_appearance, state)
        thought, memory, affect = self._recur(inputs, state.memory, state.affect, retrieved)
        logits = self.message_head(memory) if communicating else self.action_head(memory)
        entities = state.entities
        if self.entity_memory_dim:
            value = self.entity_update(torch.cat((inputs, thought, memory, affect)), retrieved)
            entities = self._store_entity(observation.partner_appearance, value, entities)
        return logits, PolicyState(memory, affect, entities)

    @staticmethod
    def _store_entity(appearance, value, entities):
        entry = EntityMemoryEntry(appearance.detach().clone(), value)
        # Copy-on-write preserves earlier rollout states and other agents.
        for index, previous in enumerate(entities):
            if torch.equal(previous.appearance, entry.appearance):
                return entities[:index] + (entry,) + entities[index + 1:]
        return entities + (entry,)

    def complete_encounter(self, experience: EncounterExperience,
                           state: PolicyState) -> PolicyState:
        """Write resolved experience without sampling or advancing Working Memory."""
        if not self.entity_memory_dim:
            return state
        observation = experience.observation
        inputs = self._encode(observation, communicating=False)
        outcome = inputs.new_tensor([
            experience.action is Action.NOTHING, experience.action is Action.GIVE,
            experience.gave, experience.received,
        ])
        value = self.encounter_update(
            torch.cat((inputs, state.memory, state.affect, outcome)),
            self.retrieve_entity(observation.partner_appearance, state))
        entities = self._store_entity(observation.partner_appearance, value, state.entities)
        return PolicyState(state.memory, state.affect, entities)


class AgentPolicy:
    """One adapter per agent; multiple adapters may reference the same network.

    Samples fixed-length messages and actions, retaining differentiable log
    probabilities, values and entropies for external training losses. Reset at episode
    boundaries; detach between truncated training segments/optimizer updates.
    Move the network to its device before constructing adapters.
    """

    def __init__(self, network: RecurrentPolicy):
        self.network = network
        self.reset()

    def reset(self) -> None:
        self.state = self.network.initial_state()
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.entropies: list[torch.Tensor] = []

    def detach(self) -> None:
        self.state = PolicyState(
            self.state.memory.detach(), self.state.affect.detach(),
            tuple(EntityMemoryEntry(entry.appearance, entry.value.detach())
                  for entry in self.state.entities))
        self.clear_decisions()

    def clear_decisions(self) -> None:
        """Release captured statistics without changing recurrent state."""
        self.log_probs.clear()
        self.values.clear()
        self.entropies.clear()

    def communicate(self, observation: Observation) -> tuple[int, ...]:
        logits, self.state = self.network(observation, self.state, communicating=True)
        if self.network.max_message_length == 0:
            return ()
        distribution = Categorical(logits=logits)
        value = self.network.value_head(self.state.memory).squeeze(-1)
        tokens = distribution.sample((self.network.max_message_length,))
        self.log_probs.append(distribution.log_prob(tokens).sum())
        self.values.append(value)
        self.entropies.append(distribution.entropy() * self.network.max_message_length)
        return tuple(tokens.tolist())

    def complete_encounter(self, experience: EncounterExperience) -> None:
        self.state = self.network.complete_encounter(experience, self.state)

    def act(self, observation: Observation) -> Action:
        logits, self.state = self.network(observation, self.state, communicating=False)
        distribution = Categorical(logits=logits)
        value = self.network.value_head(self.state.memory).squeeze(-1)
        action = distribution.sample()
        self.log_probs.append(distribution.log_prob(action))
        self.values.append(value)
        self.entropies.append(distribution.entropy())
        return (Action.NOTHING, Action.GIVE)[action.item()]
