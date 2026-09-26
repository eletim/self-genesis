import unittest

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterProtocol, Observation
from self_genesis.policy import AgentPolicy, PolicyState, RecurrentPolicy
from self_genesis.world import Action, World


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.network = RecurrentPolicy(3)
        self.observation = Observation(5, 3, 4, 2, torch.ones(3), True, (), None)

    def test_recurrence_and_affect_feedback(self):
        initial = self.network.initial_state()
        logits, state = self.network(self.observation, initial, communicating=True)
        self.assertEqual(logits.shape, (4,))
        state.memory.retain_grad()
        state.affect.retain_grad()
        subsequent, next_state = self.network(self.observation, state, communicating=False)
        changed, changed_state = self.network(
            self.observation, PolicyState(state.memory, state.affect + 1),
            communicating=False)
        self.assertFalse(torch.allclose(subsequent, changed))
        self.assertFalse(torch.allclose(next_state.memory, changed_state.memory))
        subsequent.sum().backward()
        for value in (state.memory, state.affect):
            self.assertGreater(value.grad.abs().sum().item(), 0)
        fresh, _ = self.network(self.observation, initial, communicating=False)
        self.assertFalse(torch.allclose(subsequent, fresh))

    def test_shared_weights_independent_state_and_lifecycle(self):
        first, second = AgentPolicy(self.network), AgentPolicy(self.network)
        self.assertIs(first.network, second.network)
        self.assertNotEqual(first.state.memory.data_ptr(), second.state.memory.data_ptr())
        self.assertNotEqual(first.state.affect.data_ptr(), second.state.affect.data_ptr())
        untouched = second.state
        first.communicate(self.observation)
        self.assertIs(second.state, untouched)
        self.assertEqual(second.log_probs, [])
        saved = first.state.memory.clone()
        first.detach()
        self.assertIsNone(first.state.memory.grad_fn)
        self.assertIsNone(first.state.affect.grad_fn)
        self.assertTrue(torch.equal(saved, first.state.memory))
        self.assertEqual(first.log_probs, [])
        first.act(self.observation)
        first.reset()
        self.assertEqual(first.state.memory.count_nonzero(), 0)
        self.assertEqual(first.state.affect.count_nonzero(), 0)
        self.assertEqual(first.log_probs, [])

    def test_survival_reward_can_train_all_components(self):
        world = World(ExperimentConfig(num_agents=2, appearance_dim=3,
                                       initial_life=3, initial_points=2))
        protocol = EncounterProtocol(world, seed=4)
        agents = [AgentPolicy(self.network) for _ in range(2)]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)
        before = {name: value.detach().clone() for name, value in self.network.named_parameters()}
        # Undiscounted per-agent reward-to-go, including steps without encounters.
        trajectory = []
        while True:
            starts = [len(agent.log_probs) for agent in agents]
            result = protocol.step(agents)
            trajectory.append((result.reward, [agent.log_probs[start:]
                               for agent, start in zip(agents, starts)]))
            if result.done:
                break
        returns = torch.zeros(2)
        terms = []
        for reward, decisions in reversed(trajectory):
            returns = returns + reward
            for index, log_probs in enumerate(decisions):
                terms.extend(-log_prob * returns[index] for log_prob in log_probs)
        loss = torch.stack(terms).sum()
        optimizer.zero_grad()
        loss.backward()
        for name, parameter in self.network.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        optimizer.step()
        for name, parameter in self.network.named_parameters():
            self.assertFalse(torch.equal(before[name], parameter), name)

    def test_channel_bounds_and_device(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            for length in (0, 1, 3):
                network = RecurrentPolicy(3, vocabulary_size=2, max_message_length=length).to(device)
                agent = AgentPolicy(network)
                message = agent.communicate(self.observation)
                self.assertEqual(len(message), length)
                self.assertTrue(all(type(token) is int and 0 <= token < 2 for token in message))
                self.assertIn(agent.act(self.observation), (Action.GIVE, Action.NOTHING))
                self.assertEqual(agent.state.memory.device.type, device)
                self.assertEqual(len(agent.log_probs), 1 if length == 0 else 2)

    def test_dimensions_and_incoming_messages_are_validated(self):
        for kwargs in ({"memory_dim": 0}, {"affect_dim": True},
                       {"vocabulary_size": -1}, {"max_message_length": 1.5}):
            with self.assertRaises(ValueError):
                RecurrentPolicy(3, **kwargs)
        for message in ((4,), (-1,), (True,), (0, 1, 2, 3)):
            observation = Observation(5, 3, 4, 2, torch.ones(3), True, message, None)
            with self.assertRaises(ValueError):
                AgentPolicy(self.network).communicate(observation)


if __name__ == "__main__":
    unittest.main()
