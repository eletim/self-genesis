"""Representative diagnostics are observational and agree with policy readouts."""

from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch

from self_genesis.batched_policy import BatchedObservation
from self_genesis.encounter import Observation
from self_genesis.policy import AgentPolicy, RecurrentPolicy
from self_genesis.thought_diagnostics import probe_thought_steps, probe_thought_steps_batch


class ThoughtDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.network = RecurrentPolicy(2, memory_dim=4)
        self.observation = Observation(1, 2, 3, 4, torch.ones(2), True, (), None)

    def test_metrics_and_final_readout(self):
        network = self.network
        with torch.no_grad():
            network.thought.weight.zero_()
            network.thought.weight[:, -4:] = torch.eye(4) * .5
            network.thought.bias.fill_(1)
        state = network.initial_state()
        records = probe_thought_steps(network, self.observation, state)
        self.assertEqual(len(records), 16)
        for i, record in enumerate(records, 1):
            self.assertEqual(record.step, i)
            self.assertAlmostEqual(record.thought_norm, 4 * (1 - .5 ** i))
            self.assertAlmostEqual(record.change_norm, 2 * .5 ** (i - 1))
            self.assertAlmostEqual(record.cosine_similarity, 0 if i == 1 else 1)
            self.assertEqual(record.saturation_fraction, 0)
            self.assertEqual(record.converged, record.relative_change <= 1e-3)
            if i > 1:
                prior = records[i - 2]
                self.assertAlmostEqual(record.value_change, record.value - prior.value)
                torch.testing.assert_close(torch.tensor(record.action_logit_changes),
                                           torch.tensor(record.action_logits) -
                                           torch.tensor(prior.action_logits))
        self.assertTrue(records[-1].converged)
        logits, final = network(self.observation, state, communicating=False)
        torch.testing.assert_close(torch.tensor(records[-1].action_logits), logits)
        self.assertAlmostEqual(records[-1].value, network.value_head(final.memory).item())
        baseline = network.memory_update(torch.cat((torch.zeros(4), state.affect)), state.memory)
        self.assertAlmostEqual(records[0].value_change,
                               records[0].value - network.value_head(baseline).item())

    def test_probe_preserves_state_rng_decisions_and_gradients(self):
        adapter = AgentPolicy(self.network)
        adapter.act(self.observation)
        state = adapter.state
        before = [tensor.clone() for tensor in (state.memory, state.affect, state.entities[0].value)]
        decisions = tuple(adapter.log_probs)
        rng = torch.random.get_rng_state().clone()
        with patch.object(self.network.entity_update, 'forward',
                          wraps=self.network.entity_update.forward) as write:
            records = probe_thought_steps(self.network, self.observation, state)
            write.assert_not_called()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertIs(adapter.state, state)
        self.assertEqual(tuple(adapter.log_probs), decisions)
        for old, new in zip(before, (state.memory, state.affect, state.entities[0].value)):
            torch.testing.assert_close(old, new)
        self.assertTrue(all(not isinstance(value, torch.Tensor)
                            for record in records for value in asdict(record).values()))
        self.assertTrue(all(p.grad is None for p in self.network.parameters()))
        adapter.log_probs[0].backward()
        self.assertIsNotNone(self.network.thought.weight.grad)

    def test_selected_batch_row_matches_scalar_without_full_batch_probe(self):
        network = self.network
        state = network.initial_batch_state(3, 4, slots=2)
        observation = BatchedObservation(
            torch.tensor([1., 2., 3., 4.]).expand(3, 4, 4), torch.ones(3, 4, 2),
            torch.ones(3, 4, dtype=torch.bool), torch.full((3, 4, 3), 4),
            torch.zeros(3, 4, dtype=torch.long))
        active = torch.ones(3, 4, dtype=torch.bool)
        shapes = []
        handle = network.thought.register_forward_pre_hook(
            lambda module, args: shapes.append(args[0].shape))
        snapshot = state.entities.values.clone()
        records = probe_thought_steps_batch(network, observation, state, active=active, sample=(2, 1))
        handle.remove()
        self.assertEqual(len(shapes), 16)
        self.assertTrue(all(len(shape) == 1 for shape in shapes))
        self.assertEqual(records, probe_thought_steps(network, self.observation, network.initial_state()))
        torch.testing.assert_close(state.entities.values, snapshot)
        logits, values, _ = network.forward_batch(observation, state, active=active, communicating=False)
        torch.testing.assert_close(torch.tensor(records[-1].action_logits), logits[2, 1])
        self.assertAlmostEqual(records[-1].value, values[2, 1].item(), places=6)
        active[2, 1] = False
        with self.assertRaisesRegex(ValueError, 'active'):
            probe_thought_steps_batch(network, observation, state, active=active, sample=(2, 1))
        with self.assertRaisesRegex(ValueError, 'sample'):
            probe_thought_steps_batch(network, observation, state, active=active, sample=(-1, 0))

    def test_autocast_probe_matches_final_value_and_preserves_rng(self):
        for device in ('cpu', 'cuda'):
            if device == 'cuda' and not torch.cuda.is_available():
                continue
            network = self.network.to(device)
            state = network.initial_state()
            rng = (torch.cuda.get_rng_state() if device == 'cuda'
                   else torch.random.get_rng_state()).clone()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                records = probe_thought_steps(network, self.observation, state)
                logits, final = network(self.observation, state, communicating=False)
                with torch.autocast(device_type=device, enabled=False):
                    value = network.value_head(final.memory.float()).item()
            torch.testing.assert_close(torch.tensor(records[-1].action_logits), logits.cpu().float())
            self.assertAlmostEqual(records[-1].value, value)
            after = (torch.cuda.get_rng_state() if device == 'cuda'
                     else torch.random.get_rng_state())
            self.assertTrue(torch.equal(rng, after))

    def test_zero_and_shallow_saturation_and_invalid_tolerance(self):
        for mode in ('recurrent', 'shallow'):
            network = RecurrentPolicy(2, thought_mode=mode)
            with torch.no_grad():
                network.thought.weight.zero_()
                network.thought.bias.fill_(-10)
            records = probe_thought_steps(network, self.observation, network.initial_state(),
                                          communicating=True)
            self.assertEqual(len(records), 16 if mode == 'recurrent' else 1)
            self.assertTrue(all(r.saturation_fraction == 1 for r in records))
            if mode == 'recurrent':
                self.assertTrue(all(r.converged and r.cosine_similarity == 0 for r in records))
        for tolerance in (-1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'convergence_tolerance'):
                probe_thought_steps(network, self.observation, network.initial_state(),
                                    convergence_tolerance=tolerance)


if __name__ == '__main__':
    unittest.main()
