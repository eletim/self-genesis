"""Ordered encounters across independent worlds, without per-agent callbacks."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from torch.distributions import Categorical

from self_genesis.batched_random import BatchedRandom
from self_genesis.batched_policy import BatchedObservation, BatchedPolicyState
from self_genesis.batched_world import BatchedStepResult, BatchedWorld
from self_genesis.policy import RecurrentPolicy


@dataclass(frozen=True)
class BatchedDecision:
    """One phase's choices and differentiable statistics, routed [world, agent].

    Message choices end in ordered token slots; action choices are 0 NOTHING,
    1 GIVE. Only active entries are decisions. An empty channel still advances
    recurrence, but has no sampled decision or associated statistics.
    """

    observation: BatchedObservation
    active: torch.Tensor
    communicating: bool
    choice: torch.Tensor
    log_prob: torch.Tensor | None
    value: torch.Tensor | None
    entropy: torch.Tensor | None


@dataclass(frozen=True)
class BatchedEncounterResult:
    world: BatchedStepResult
    state: BatchedPolicyState
    # [world, 2], ordered first/second; -1 for rows without an encounter.
    pairs: torch.Tensor
    decisions: tuple[BatchedDecision, ...]


class BatchedEncounterProtocol:
    """Run message, reply, first action, second action, then local completion.

    State is supplied and returned explicitly, retaining recurrent graphs. The
    caller owns episode resets and optimizer detach boundaries. Each world's
    isolated Philox stream supplies uniforms on the world device, alongside
    selection, routing and policy execution. Streams persist across world/state resets.
    Seeds reproduce this executor, not the legacy Python random.sample stream.
    """

    def __init__(self, world: BatchedWorld, network: RecurrentPolicy, *,
                 seeds: Sequence[int]):
        seeds = tuple(seeds)
        if (len(seeds) != world.state.life.shape[0]
                or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)):
            raise ValueError("Provide one integer seed in [0, 2**63) per world")
        parameter = next(network.parameters())
        if (network.appearance_dim != world.config.appearance_dim
                or parameter.device != world.state.life.device):
            raise ValueError("Policy and world must share Appearance dimension and device")
        self.world = world
        self.network = network
        self._random = BatchedRandom(seeds, device=world.state.life.device)

    def _uniforms(self, active):
        return self._random.uniform(4 + 2 * self.network.max_message_length, active)

    @staticmethod
    def _select(living, uniforms):
        """Uniform living ranks give an ordered pair without replacement."""
        count = living.sum(dim=1)
        rank = (uniforms[:, 0] * count).long() + 1
        first = (living & (living.cumsum(dim=1) == rank[:, None])).long().argmax(dim=1)
        remaining = living.clone()
        remaining.scatter_(1, first[:, None], False)
        rank = (uniforms[:, 1] * (count - 1).clamp(min=0)).long() + 1
        second = (remaining & (remaining.cumsum(dim=1) == rank[:, None])).long().argmax(dim=1)
        return torch.stack((first, second), dim=1)

    def _decide(self, observation, state, active, *, communicating, uniforms):
        logits, values, state = self.network.forward_batch(
            observation, state, communicating=communicating, active=active)
        length = self.network.max_message_length if communicating else 1
        choice = torch.full((*active.shape, length),
                            self.network.vocabulary_size if communicating else 0,
                            device=active.device, dtype=torch.int64)
        log_prob = entropy = value = None
        if length:
            distribution = Categorical(logits=logits[active])
            draws = uniforms[:, None, :].expand(*active.shape, length)[active]
            cumulative = distribution.probs.cumsum(dim=-1)
            samples = (draws[..., None] >= cumulative[:, None, :]).sum(dim=-1)
            samples = samples.clamp(max=logits.shape[-1] - 1)
            choice[active] = samples
            log_prob = values.new_zeros(active.shape)
            entropy = values.new_zeros(active.shape)
            log_prob[active] = distribution.log_prob(samples.T).sum(dim=0)
            entropy[active] = distribution.entropy() * length
            value = values
        return BatchedDecision(observation, active, communicating,
                               choice if communicating else choice.squeeze(-1),
                               log_prob, value, entropy), state

    def step(self, state: BatchedPolicyState) -> BatchedEncounterResult:
        world = self.world
        eligible = ~world.done & (world.alive.sum(dim=1) >= 2)
        uniforms = self._uniforms(eligible)
        pairs = self._select(world.alive, uniforms)
        agents = torch.arange(world.state.life.shape[1], device=world.state.life.device)
        first = eligible[:, None] & (agents == pairs[:, :1])
        second = eligible[:, None] & (agents == pairs[:, 1:])
        participants = first | second
        targets = torch.full_like(world.state.life, -1)
        if not bool(eligible.any()):
            return BatchedEncounterResult(world.step(targets), state,
                                          torch.full_like(pairs, -1), ())
        partner = torch.where(first, pairs[:, 1:], pairs[:, :1]).expand_as(targets)
        parameter = next(self.network.parameters())
        resources = torch.stack((world.state.life, world.state.points,
                                 world.state.life.gather(1, partner),
                                 world.state.points.gather(1, partner)), dim=-1)
        appearance = world.state.appearance.gather(
            1, partner[..., None].expand(*partner.shape, self.network.appearance_dim)).to(parameter)
        length = self.network.max_message_length
        empty = torch.full((*targets.shape, length), self.network.vocabulary_size,
                           device=targets.device, dtype=torch.int64)
        observation = BatchedObservation(resources, appearance, first, empty,
                                         torch.zeros_like(targets))
        message, state = self._decide(observation, state, first, communicating=True,
                                      uniforms=uniforms[:, 2:2 + length])
        incoming = message.choice.gather(1, partner[..., None].expand_as(empty))
        reply_observation = replace(observation, received_message=incoming)
        reply, state = self._decide(reply_observation, state, second, communicating=True,
                                    uniforms=uniforms[:, 2 + length:2 + 2 * length])
        incoming = torch.where(first[..., None], reply.choice.gather(
            1, partner[..., None].expand_as(empty)), incoming)
        first_observation = replace(observation, received_message=incoming)
        action1, state = self._decide(first_observation, state, first, communicating=False,
                                      uniforms=uniforms[:, -2:-1])
        visible_action = torch.where(second, action1.choice.gather(1, partner) + 1, 0)
        second_observation = replace(first_observation, partner_action=visible_action)
        action2, state = self._decide(second_observation, state, second, communicating=False,
                                      uniforms=uniforms[:, -1:])
        actions = torch.where(first, action1.choice, action2.choice)
        targets = torch.where(participants & (actions == 1), partner, targets)
        result = world.step(targets)
        # Retain pre-step resources/messages/Appearance, revealing final actions
        # and successful transfers only to the two participants, even if they die.
        completed = replace(first_observation, partner_action=actions.gather(1, partner) + 1)
        state = self.network.complete_encounter_batch(
            completed, state, action=actions, gave=result.successful_transfers,
            received=result.successful_transfers.gather(1, partner), active=participants)
        return BatchedEncounterResult(
            result, state, torch.where(eligible[:, None], pairs, -1),
            (message, reply, action1, action2))
