"""Bounded collection of separate, differentiable agent trajectories."""

from dataclasses import dataclass

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterProtocol, Observation
from self_genesis.observation import RunRecorder
from self_genesis.policy import AgentPolicy, PolicyState, RecurrentPolicy
from self_genesis.world import Action, World


@dataclass(frozen=True)
class PolicyDecision:
    """Callback data; unsampled empty messages have no log_prob, value or entropy."""

    observation: Observation
    choice: tuple[int, ...] | Action
    log_prob: torch.Tensor | None
    state_before: PolicyState
    state_after: PolicyState
    value: torch.Tensor | None
    entropy: torch.Tensor | None


@dataclass(frozen=True)
class AgentExperience:
    step: int
    reward: float
    terminated: bool
    decisions: tuple[PolicyDecision, ...]


@dataclass(frozen=True)
class Rollout:
    # Tuple position routes experience to an agent; it is never a policy input.
    experiences: tuple[tuple[AgentExperience, ...], ...]
    steps: int
    terminated: bool
    truncated: bool
    horizon_completed: bool = False


class _RecordingPolicy:
    """Capture callbacks without changing the policy or encounter protocol."""

    def __init__(self, agent: AgentPolicy):
        self.agent = agent
        self.decisions: list[PolicyDecision] = []

    def _record(self, observation: Observation, *, communicating: bool):
        before = self.agent.state
        start = len(self.agent.log_probs)
        choice = (self.agent.communicate(observation) if communicating
                  else self.agent.act(observation))
        sampled = len(self.agent.log_probs) > start
        log_prob = self.agent.log_probs[-1] if sampled else None
        value = self.agent.values[-1] if sampled else None
        entropy = self.agent.entropies[-1] if sampled else None
        self.decisions.append(PolicyDecision(
            observation, choice, log_prob, before, self.agent.state, value, entropy))
        return choice

    def communicate(self, observation: Observation) -> tuple[int, ...]:
        return self._record(observation, communicating=True)

    def complete_encounter(self, experience) -> None:
        self.agent.complete_encounter(experience)

    def act(self, observation: Observation) -> Action:
        return self._record(observation, communicating=False)


class RolloutCollector:
    """Own one world and independent adapters around shared policy weights.

    collect() stops at extinction, the survival horizon, or its step budget.
    A budget boundary before either ending is a truncation: another collect()
    continues the same episode and recurrent graph. After horizon completion,
    collect() returns an empty completed segment until reset(). Consume losses before detach() and
    optimizer updates. reset() explicitly starts the configured episode anew.
    Returned records own their decision references and survive reset/detach.
    """

    def __init__(self, config: ExperimentConfig, network: RecurrentPolicy, *,
                 recorder: RunRecorder | None = None):
        if network.appearance_dim != config.appearance_dim:
            raise ValueError("Policy and world appearance dimensions must match")
        self.recorder = recorder
        self.config = config
        self.network = network
        self.agents = tuple(AgentPolicy(network) for _ in range(config.num_agents))
        self.world = World(self.config)
        self.protocol = EncounterProtocol(
            self.world, seed=self.config.seed,
            vocabulary_size=self.network.vocabulary_size,
            max_message_length=self.network.max_message_length)
        self.elapsed_steps = 0
        if self.recorder is not None:
            self.recorder.start_episode(self)

    def reset(self) -> None:
        """Restore the world and agent state while preserving sampling streams."""
        generation_rng = self.world._generation_rng
        self.world = World(self.config, seed_rng=False)
        self.world._generation_rng = generation_rng
        self.protocol.world = self.world
        self.elapsed_steps = 0
        for agent in self.agents:
            agent.reset()
        if self.recorder is not None:
            self.recorder.start_episode(self)

    def detach(self) -> None:
        """Keep recurrent values but cut history before the next training segment."""
        for agent in self.agents:
            agent.detach()

    def collect(self, max_steps: int) -> Rollout:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        experiences: list[list[AgentExperience]] = [[] for _ in self.agents]
        steps = 0
        horizon = self.config.survival_horizon
        while (steps < max_steps and bool(self.world.alive.any())
               and (horizon is None or self.elapsed_steps < horizon)):
            alive = self.world.alive.tolist()
            policies = [_RecordingPolicy(agent) for agent in self.agents]
            result = self.protocol.step(policies)
            if self.recorder is not None:
                self.recorder.record_step(self, result, policies)
            for index, (agent, policy) in enumerate(zip(self.agents, policies)):
                if alive[index]:
                    experiences[index].append(AgentExperience(
                        self.elapsed_steps, float(result.reward[index]),
                        bool(result.died[index]), tuple(policy.decisions)))
                # The returned records retain the graph. Do not accumulate a
                # second unbounded history in the adapters across collections.
                agent.clear_decisions()
                if result.died[index]:
                    agent.reset()
            self.elapsed_steps += 1
            steps += 1
        terminated = not bool(self.world.alive.any())
        horizon_completed = (not terminated and horizon is not None
                             and self.elapsed_steps >= horizon)
        if self.recorder is not None:
            self.recorder.record_summary(
                self, max_steps=max_steps, steps=steps, terminated=terminated,
                horizon_completed=horizon_completed)
        return Rollout(tuple(tuple(items) for items in experiences), steps,
                       terminated, not (terminated or horizon_completed), horizon_completed)
