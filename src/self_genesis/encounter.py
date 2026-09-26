"""Seeded encounters with an uninterpreted, bounded communication channel."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class EncounterExperience:
    """Participant-visible decisions and gift outcomes, without world identity.

    Observation retains pre-step resources and the Appearance actually seen,
    with the partner's final action revealed after both decisions. Transfer
    success is visible to donor and recipient; generation draws are not exposed.
    """

    observation: Observation
    action: Action
    gave: bool
    received: bool


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

    def _observe(self, agent, partner, *, first, message=(), action=None):
        state = self.world.state
        return Observation(
            life=int(state.life[agent]), points=int(state.points[agent]),
            partner_life=int(state.life[partner]),
            partner_points=int(state.points[partner]),
            partner_appearance=state.appearance[partner].clone(),
            first=first, received_message=message, partner_action=action,
        )

    def step(self, policies: Sequence[Policy]) -> StepResult:
        """Run communication and decisions before advancing survival time.

        Empty messages are allowed. Tokens have no assigned meaning or effect
        on rewards. Invalid messages/actions leave world state unchanged.
        With fewer than two survivors, time advances without policy calls.
        After resolution, participants with a complete_encounter callback receive
        local experience, including the second action and successful gifts.
        """
        state = self.world.state
        if len(policies) != state.life.numel():
            raise ValueError("Provide exactly one policy per agent")
        decisions = [Decision() for _ in policies]
        living = self.world.alive.nonzero().flatten().tolist()
        if len(living) >= 2:
            first, second = self._random.sample(living, 2)

            message = self._message(policies[first].communicate(
                self._observe(first, second, first=True)))
            reply = self._message(policies[second].communicate(
                self._observe(second, first, first=False, message=message)))
            first_observation = self._observe(first, second, first=True, message=reply)
            first_action = policies[first].act(first_observation)
            if not isinstance(first_action, Action):
                raise ValueError("Policy must choose an Action")
            second_observation = self._observe(
                second, first, first=False, message=message, action=first_action)
            second_action = policies[second].act(second_observation)
            if not isinstance(second_action, Action):
                raise ValueError("Policy must choose an Action")
            for agent, partner, action in (
                (first, second, first_action), (second, first, second_action),
            ):
                decisions[agent] = Decision(action, partner if action is Action.GIVE else None)
        result = self.world.step(decisions)
        if len(living) >= 2:
            for agent, partner, observation, action, partner_action in (
                (first, second, first_observation, first_action, second_action),
                (second, first, second_observation, second_action, first_action),
            ):
                # Completion is optional for fixed and external policies.
                complete = getattr(policies[agent], "complete_encounter", None)
                if complete is not None:
                    complete(EncounterExperience(
                        replace(observation, partner_action=partner_action), action,
                        (agent, partner) in result.successful_transfers,
                        (partner, agent) in result.successful_transfers))
        return result
