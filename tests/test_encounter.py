from dataclasses import fields
import random
import unittest

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterProtocol, Observation
from self_genesis.world import Action, World


class RecordingPolicy:
    def __init__(self, index, events, message=(0, 3), action=Action.NOTHING):
        self.index, self.events = index, events
        self.message, self.action = message, action

    def communicate(self, observation):
        self.events.append((self.index, "message", observation))
        return self.message

    def act(self, observation):
        self.events.append((self.index, "action", observation))
        return self.action


class EncounterTests(unittest.TestCase):
    def make_protocol(self, seed=42, count=4, **kwargs):
        world = World(ExperimentConfig(num_agents=count, initial_life=1000))
        protocol = EncounterProtocol(world, seed=seed, **kwargs)
        events = []
        policies = [RecordingPolicy(i, events) for i in range(count)]
        return world, protocol, events, policies

    def test_seeded_pairing_and_random_roles_are_isolated(self):
        def trace(seed):
            _, protocol, events, policies = self.make_protocol(seed)
            pairs = []
            for _ in range(100):
                random.seed(999)
                random.random()
                protocol.step(policies)
                pairs.append((events[-4][0], events[-3][0]))
            return pairs
        pairs = trace(42)
        self.assertEqual(pairs, trace(42))
        self.assertNotEqual(pairs, trace(43))
        self.assertEqual(set(pairs), {(a, b) for a in range(4) for b in range(4) if a != b})

    def test_order_observations_and_partner_directed_gifts(self):
        world, protocol, events, policies = self.make_protocol()
        appearance = world.state.appearance.clone()
        for policy in policies:
            policy.action = Action.GIVE
            policy.message = (policy.index,)
        result = protocol.step(policies)
        first, second = events[0][0], events[1][0]
        self.assertEqual([(i, phase) for i, phase, _ in events],
                         [(first, "message"), (second, "message"),
                          (first, "action"), (second, "action")])
        self.assertEqual(events[0][2].received_message, ())
        for i, _, observation in events[1:]:
            partner = second if i == first else first
            self.assertEqual(observation.received_message, (partner,))
        self.assertIsNone(events[2][2].partner_action)
        self.assertEqual(events[3][2].partner_action, Action.GIVE)
        for i, _, observation in events:
            partner = second if i == first else first
            self.assertEqual(observation.first, i == first)
            self.assertEqual((observation.life, observation.points,
                              observation.partner_life, observation.partner_points),
                             (1000, 3, 1000, 3))
            self.assertTrue(torch.equal(observation.partner_appearance, appearance[partner]))
            observation.partner_appearance.zero_()
        self.assertTrue(torch.equal(world.state.appearance, appearance))
        self.assertEqual({field.name for field in fields(Observation)},
                         {"life", "points", "partner_life", "partner_points",
                          "partner_appearance", "first", "received_message", "partner_action"})
        self.assertEqual(world.state.life.tolist(),
                         [1000 if i in (first, second) else 999 for i in range(4)])
        self.assertEqual(world.state.points.tolist(),
                         [2 if i in (first, second) else 3 for i in range(4)])
        self.assertEqual(result.reward.tolist(), [1] * 4)

    def test_dead_agents_excluded_and_single_survivor_decays(self):
        world, protocol, events, policies = self.make_protocol()
        world.state.life[:] = torch.tensor([0, 2, 0, 2])
        protocol.step(policies)
        self.assertEqual({event[0] for event in events}, {1, 3})
        world.state.life[1] = 0
        events.clear()
        self.assertTrue(protocol.step(policies).done)
        self.assertEqual(events, [])
        self.assertEqual(protocol.step(policies).reward.tolist(), [0] * 4)

    def test_invalid_messages_actions_and_policy_counts_are_atomic(self):
        for message, action in (([0, 1, 2, 3], Action.NOTHING),
                                ([-1], Action.NOTHING), ([4], Action.NOTHING),
                                ([True], Action.NOTHING), ([1.0], Action.NOTHING),
                                (None, Action.NOTHING), ("0", Action.NOTHING),
                                ((), "GIVE")):
            with self.subTest(message=message, action=action):
                world, protocol, _, policies = self.make_protocol()
                # Fail on the second participant, after the first has run.
                second = random.Random(42).sample(range(4), 2)[1]
                policies[second].message, policies[second].action = message, action
                with self.assertRaises(ValueError):
                    protocol.step(policies)
                self.assertEqual(world.state.life.tolist(), [1000] * 4)
                self.assertEqual(world.state.points.tolist(), [3] * 4)
        with self.assertRaises(ValueError):
            protocol.step(policies[:-1])

    def test_channel_bounds_and_tokens_have_no_world_effect(self):
        for message in ((), (0,), (3, 2, 1)):
            world, protocol, _, policies = self.make_protocol(count=2)
            for policy in policies:
                policy.message = message
            protocol.step(policies)
            self.assertEqual(world.state.life.tolist(), [999, 999])
            self.assertEqual(world.state.points.tolist(), [3, 3])
        world, protocol, _, policies = self.make_protocol(count=2, max_message_length=0)
        for policy in policies:
            policy.message = ()
        protocol.step(policies)
        for kwargs in ({"vocabulary_size": 0}, {"vocabulary_size": True},
                       {"max_message_length": -1}, {"max_message_length": 1.5},
                       {"seed": -1}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EncounterProtocol(world, **({"seed": 0} | kwargs))


if __name__ == "__main__":
    unittest.main()
