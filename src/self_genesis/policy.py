"""Trainable recurrent policy, with state owned by individual agents."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from self_genesis.encounter import Observation
from self_genesis.world import Action


@dataclass(frozen=True)
class PolicyState:
    memory: torch.Tensor
    affect: torch.Tensor


class RecurrentPolicy(nn.Module):
    """Shared weights only; no individual state or identity labels.

    Previous memory and unlabeled affect feed thought and the next memory
    update. New affect depends on the observation, thought and updated memory,
    and feeds the following callback. Message slots have no assigned meaning.
    """

    def __init__(self, appearance_dim: int, *, vocabulary_size: int = 4,
                 max_message_length: int = 3, memory_dim: int = 16,
                 affect_dim: int = 4):
        super().__init__()
        for name, value, minimum in (
            ("appearance_dim", appearance_dim, 1),
            ("vocabulary_size", vocabulary_size, 1),
            ("max_message_length", max_message_length, 0),
            ("memory_dim", memory_dim, 1), ("affect_dim", affect_dim, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.appearance_dim = appearance_dim
        self.vocabulary_size = vocabulary_size
        self.max_message_length = max_message_length
        self.memory_dim = memory_dim
        self.affect_dim = affect_dim
        # Four resources, role, three partner-action categories, callback phase,
        # appearance, and ordered message slots (including an empty category).
        input_dim = 9 + appearance_dim + max_message_length * (vocabulary_size + 1)
        self.thought = nn.Linear(input_dim + memory_dim + affect_dim, memory_dim)
        self.memory_update = nn.GRUCell(memory_dim + affect_dim, memory_dim)
        self.affect_update = nn.Linear(input_dim + 2 * memory_dim, affect_dim)
        self.message_head = nn.Linear(memory_dim, vocabulary_size)
        self.action_head = nn.Linear(memory_dim, 2)
        self.value_head = nn.Linear(memory_dim, 1)

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

    def forward(self, observation: Observation, state: PolicyState, *,
                communicating: bool) -> tuple[torch.Tensor, PolicyState]:
        inputs = self._encode(observation, communicating)
        thought = torch.tanh(self.thought(torch.cat((inputs, state.memory, state.affect))))
        memory = self.memory_update(torch.cat((thought, state.affect)), state.memory)
        affect = torch.tanh(self.affect_update(torch.cat((inputs, thought, memory))))
        logits = self.message_head(memory) if communicating else self.action_head(memory)
        return logits, PolicyState(memory, affect)


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
        self.state = PolicyState(self.state.memory.detach(), self.state.affect.detach())
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

    def act(self, observation: Observation) -> Action:
        logits, self.state = self.network(observation, self.state, communicating=False)
        distribution = Categorical(logits=logits)
        value = self.network.value_head(self.state.memory).squeeze(-1)
        action = distribution.sample()
        self.log_probs.append(distribution.log_prob(action))
        self.values.append(value)
        self.entropies.append(distribution.entropy())
        return (Action.NOTHING, Action.GIVE)[action.item()]
