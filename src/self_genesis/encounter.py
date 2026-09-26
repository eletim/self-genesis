"""Seeded encounters with an uninterpreted, bounded communication channel."""

from collections.abc import Sequence
from dataclasses import dataclass
import random
from typing import Protocol

import torch

from self_genesis.world import Action, Decision, StepResult, World


@dataclass(frozen=True)
class Observation:
    life: int
    points: int
    partner_life: int
    partner_points: int
    partner_appearance: torch.Tensor
    first: bool
    received_message: tuple[int, ...]
    partner_action: Action | None


class Policy(Protocol):
    def communicate(self, observation: Observation) -> Sequence[int]: ...

    def act(self, observation: Observation) -> Action: ...


class EncounterProtocol:
    """Sample one ordered living pair per world step using an isolated RNG.

    First sends a message, then second replies. Both then choose an action,
    first before second. Second can observe first's chosen action. Gifts and
    time resolve together through World.step; everyone else chooses NOTHING.
    Policy positions are only used internally for routing, never observations.
    """

    def __init__(self, world: World, *, seed: int, vocabulary_size: int = 4,
                 max_message_length: int = 3):
        for name, value, minimum in (
            ("seed", seed, 0), ("vocabulary_size", vocabulary_size, 1),
            ("max_message_length", max_message_length, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.world = world
        self.vocabulary_size = vocabulary_size
        self.max_message_length = max_message_length
        self._random = random.Random(seed)

    def _message(self, message: Sequence[int]) -> tuple[int, ...]:
        if not isinstance(message, Sequence) or len(message) > self.max_message_length:
            raise ValueError("Message must be a sequence within max_message_length")
        if any(type(token) is not int or not 0 <= token < self.vocabulary_size
               for token in message):
            raise ValueError("Message tokens must be integers within vocabulary_size")
        return tuple(message)

    def step(self, policies: Sequence[Policy]) -> StepResult:
        """Run communication and decisions before advancing survival time.

        Empty messages are allowed. Tokens have no assigned meaning or effect
        on rewards. Invalid messages/actions leave world state unchanged.
        With fewer than two survivors, time advances without policy calls.
        """
        state = self.world.state
        if len(policies) != state.life.numel():
            raise ValueError("Provide exactly one policy per agent")
        decisions = [Decision() for _ in policies]
        living = self.world.alive.nonzero().flatten().tolist()
        if len(living) >= 2:
            first, second = self._random.sample(living, 2)

            def observe(agent, partner, message=(), action=None):
                return Observation(
                    life=int(state.life[agent]), points=int(state.points[agent]),
                    partner_life=int(state.life[partner]),
                    partner_points=int(state.points[partner]),
                    partner_appearance=state.appearance[partner].clone(),
                    first=agent == first, received_message=message,
                    partner_action=action,
                )

            message = self._message(policies[first].communicate(observe(first, second)))
            reply = self._message(policies[second].communicate(
                observe(second, first, message)))
            first_action = policies[first].act(observe(first, second, reply))
            if not isinstance(first_action, Action):
                raise ValueError("Policy must choose an Action")
            second_action = policies[second].act(
                observe(second, first, message, first_action))
            if not isinstance(second_action, Action):
                raise ValueError("Policy must choose an Action")
            for agent, partner, action in (
                (first, second, first_action), (second, first, second_action),
            ):
                decisions[agent] = Decision(action, partner if action is Action.GIVE else None)
        return self.world.step(decisions)
