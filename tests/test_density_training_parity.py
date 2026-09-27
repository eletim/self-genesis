"""Controlled multi-encounter episodes through training and trace consumers."""

from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from examples.benchmark_rtx5090 import encounter_count
from self_genesis.batched_rollout import BatchedRolloutCollector
from self_genesis.config import ExperimentConfig
from self_genesis.experiment import ExperimentState
from self_genesis.observation import BatchedTraceRecorder, RunRecorder
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import survival_loss_components, train_batch
from self_genesis.world import Action


class DensityTrainingParityTests(unittest.TestCase):
    def test_controlled_multi_pair_training_traces_and_memory(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            for length in (0, 3):
                for density in ({'encounter_count': 2}, {'encounter_fraction': 0.8}):
                    with self.subTest(device=device, length=length, density=density):
                        self.check_episode(device, length, density)

    def check_episode(self, device, length, density):
        torch.manual_seed(19)
        config = ExperimentConfig(
            num_agents=5, appearance_dim=3, initial_life=5, initial_points=1,
            survival_horizon=4, device=device, batched=True, num_worlds=2,
            trace_worlds=(0, 1), point_generation_probability_min=1,
            point_generation_probability_max=1, **density)
        network = RecurrentPolicy(3, max_message_length=length).to(device).double()
        reference = RecurrentPolicy(3, max_message_length=length).to(device).double()
        reference.load_state_dict(network.state_dict())
        collector = BatchedRolloutCollector(config, network, seeds=[11, 22])
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        expected_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
        lives = ([1] * 5, [5, 3, 1, 4, 4])
        points = ([0] * 5, [0, 0, 0, 1, 1])
        # Independently prescribed choices: lowest living ranks in world 0,
        # highest in world 1; mutual GIVE in pair 0, GIVE/NOTHING in pair 1.
        draws = []
        for _ in range(4):
            for slot in range(2):
                draw = torch.zeros(2, 4 + 2 * length, device=device, dtype=torch.float64)
                draw[1, :2] = 1 - 1e-12
                draw[:, 2:2 + length] = slot * (1 - 1e-12)
                draw[:, 2 + length:2 + 2 * length] = (1 - slot) * (1 - 1e-12)
                draw[:, -2] = 1 - 1e-12
                draw[:, -1] = (1 - slot) * (1 - 1e-12)
                draws.append(draw)
        reset, batch_step, collect = collector.reset, collector.protocol.step, collector.collect
        results, snapshots, rollouts, traces = [], [], [], []
        collector.trace_recorder = BatchedTraceRecorder(
            config, lambda kind, **values: traces.append(values))

        def reset_resources(row, *, seed):
            reset(row, seed=seed)
            collector.world.state.life[row] = torch.tensor(lives[row], device=device)
            collector.world.state.points[row] = torch.tensor(points[row], device=device)

        def record_step(state):
            result = batch_step(state)
            results.append(result)
            snapshots.append({f.name: getattr(collector.world.state, f.name).clone()
                              for f in fields(ExperimentState)})
            return result

        def record_rollout(budget):
            rollout = collect(budget)
            rollouts.append(rollout)
            return rollout

        with patch.object(collector, 'reset', side_effect=reset_resources), patch.object(
                collector.protocol, '_uniforms', side_effect=draws), patch.object(
                collector.protocol, 'step', side_effect=record_step), patch.object(
                collector, 'collect', side_effect=record_rollout):
            actual = train_batch(collector, optimizer, seeds=[11, 22])
        self.assertEqual(actual.steps, (1, 4))
        self.assertEqual(actual.terminated, (True, False))
        self.assertEqual(actual.horizon_completed, (False, True))
        self.assertEqual([(t['world'], t['step']) for t in traces],
                         [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3)])
        self.assertEqual(results[0].world.successful_transfers[1].tolist(),
                         [False, False, False, True, True])
        for result, snapshot in zip(results[1:], snapshots[1:]):
            self.assertTrue((result.pairs[0] == -1).all())
            self.assertFalse(result.world.reward[0].any())
            for name in snapshot:
                torch.testing.assert_close(snapshot[name][0], snapshots[0][name][0])
            torch.testing.assert_close(result.state.entities.values[0],
                                       results[0].state.entities.values[0])
        trace_count = len(traces)
        self.assertEqual(collector.collect(1).experiences, ())
        self.assertEqual(len(traces), trace_count)

        losses, counts = [], []
        for row, seed in enumerate((11, 22)):
            scalar = RolloutCollector(replace(config, batched=False, trace_worlds=(0,), seed=seed),
                                      reference)
            scalar.world.state.life[:] = torch.tensor(lives[row], device=device)
            scalar.world.state.points[:] = torch.tensor(points[row], device=device)
            scalar_step = scalar.protocol.step
            expected_pairs = []

            def scripted(policies):
                step = scalar.elapsed_steps
                living = scalar.world.alive.nonzero().flatten().tolist()
                if row:
                    living.reverse()
                count = config.encounter_pairs(len(living))
                selected = living[:2 * count]
                pairs = list(zip(selected[::2], selected[1::2]))
                expected_pairs.append([list(pair) for pair in pairs])
                samples = []
                for slot in range(count):
                    if length:
                        samples.extend(torch.full((length,), token, device=device, dtype=torch.long)
                                       for token in (slot * (network.vocabulary_size - 1),
                                                     (1 - slot) * (network.vocabulary_size - 1)))
                    samples.extend(torch.tensor(choice, device=device) for choice in (1, 1 - slot))
                with patch.object(scalar.protocol._random, 'sample', return_value=selected), patch(
                        'self_genesis.policy.Categorical.sample', side_effect=samples):
                    expected = scalar_step(policies)
                result = results[step]
                self.assertEqual(result.pairs[row].tolist(), selected + [-1] * (4 - len(selected)))
                for name in ('reward', 'died', 'generated_points'):
                    torch.testing.assert_close(getattr(result.world, name)[row], getattr(expected, name))
                transfers = {(donor, recipient) for a, b in pairs
                             for donor, recipient in ((a, b), (b, a))
                             if result.world.successful_transfers[row, donor]}
                self.assertEqual(transfers, set(expected.successful_transfers))
                for name, snapshot in snapshots[step].items():
                    torch.testing.assert_close(snapshot[row], getattr(scalar.world.state, name))
                for slot, pair in enumerate(pairs):
                    for phase, decision in enumerate(result.decisions):
                        agent = pair[phase % 2]
                        recorded = policies[agent].decisions[phase // 2]
                        seen, obs = recorded.observation, decision.observation
                        self.assertTrue(decision.active[row, agent])
                        self.assertEqual(obs.resources[row, agent].tolist(),
                                         [seen.life, seen.points, seen.partner_life, seen.partner_points])
                        self.assertEqual(bool(obs.first[row, agent]), seen.first)
                        self.assertEqual(tuple(t for t in obs.received_message[row, agent].tolist()
                                               if t != network.vocabulary_size), seen.received_message)
                        self.assertEqual(int(obs.partner_action[row, agent]),
                                         0 if seen.partner_action is None else
                                         1 + (seen.partner_action == Action.GIVE))
                        torch.testing.assert_close(obs.partner_appearance[row, agent],
                                                   seen.partner_appearance.to(network.thought.weight))
                        choice = (list(recorded.choice) if decision.communicating else
                                  int(recorded.choice == Action.GIVE))
                        self.assertEqual(decision.choice[row, agent].tolist(), choice)
                        if recorded.log_prob is not None:
                            for name in ('log_prob', 'value', 'entropy'):
                                torch.testing.assert_close(getattr(decision, name)[row, agent],
                                                           getattr(recorded, name))
                # Compare completion before scalar death reset. Batched dead
                # observers freeze their final state until the world resets.
                for agent, policy in enumerate(scalar.agents):
                    if not expected.reward[agent]:
                        continue
                    torch.testing.assert_close(result.state.memory[row, agent], policy.state.memory)
                    torch.testing.assert_close(result.state.affect[row, agent], policy.state.affect)
                    for partner in range(5):
                        key = collector.world.state.appearance[:, partner:partner + 1].expand(-1, 5, -1).double()
                        torch.testing.assert_close(result.state.entities.retrieve(key)[row, agent],
                                                   reference.retrieve_entity(key[row, agent], policy.state))
                return expected

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'trace.jsonl'
                with RunRecorder(path) as recorder:
                    scalar.recorder = recorder
                    recorder.start_episode(scalar)
                    with patch.object(scalar.protocol, 'step', side_effect=scripted):
                        rollout = scalar.collect(10)
                records = [json.loads(line) for line in path.read_text().splitlines()]
            scalar_traces = [r for r in records if r['type'] == 'step']
            batch_traces = [r for r in traces if r['world'] == row]
            self.assertEqual([t['pairs'] for t in scalar_traces], expected_pairs)
            self.assertEqual([t['pairs'] for t in batch_traces], expected_pairs)
            for scalar_trace, batch_trace in zip(scalar_traces, batch_traces):
                for name in ('rewards', 'died', 'generated_points', 'life', 'points'):
                    self.assertEqual(scalar_trace[name], batch_trace[name])
                participants = {a for pair in scalar_trace['pairs'] for a in pair}
                self.assertEqual({c['agent'] for c in scalar_trace['entity_memory_completions']},
                                 participants)
                self.assertEqual(len(scalar_trace['callbacks']), 2 * len(participants))
                for callback in scalar_trace['callbacks']:
                    agent = callback['agent']
                    communicating = callback['phase'] == 'message'
                    decision, = [d for d in batch_trace['decisions']
                                 if d['communicating'] == communicating and d['active'][agent]]
                    expected_choice = (callback['choice'] if communicating else
                                       int(callback['choice'] == 'GIVE'))
                    self.assertEqual(decision['choice'][agent], expected_choice)
            count = sum(map(len, expected_pairs))
            self.assertEqual(encounter_count(rollout, False), count)
            self.assertEqual(sum(records[-1]['action_counts'].values()), 2 * count)
            counts.append(count)
            self.assertEqual(tuple(sum(e.reward for e in agent) for agent in rollout.experiences),
                             actual.survival_returns[row])
            losses.append(survival_loss_components(rollout))
        self.assertEqual(encounter_count(rollouts[0], True), sum(counts))
        self.assertEqual(actual.metrics["encounters"], sum(counts))
        self.assertEqual(actual.metrics["action_callbacks"], 2 * sum(counts))
        for name in ('loss', 'actor_loss', 'value_loss', 'action_entropy', 'message_entropy'):
            expected = sum(getattr(loss, name) for loss in losses) / 2
            torch.testing.assert_close(torch.tensor(getattr(actual, name), dtype=torch.float64),
                                       expected.detach().cpu(), atol=2e-6, rtol=2e-5)
        expected_optimizer.zero_grad(set_to_none=True)
        (sum(loss.loss for loss in losses) / 2).backward()
        expected_optimizer.step()
        for (name, parameter), (_, expected) in zip(network.named_parameters(), reference.named_parameters()):
            if expected.grad is None:
                self.assertIsNone(parameter.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-6, rtol=2e-4)
            torch.testing.assert_close(parameter, expected, atol=2e-6, rtol=2e-4)

    def test_no_pair_traces_and_counters_at_death_and_horizon(self):
        for density in ({'encounter_count': 0}, {'encounter_fraction': 0.3}):
            config = ExperimentConfig(num_agents=5, initial_life=3, initial_points=0,
                                      survival_horizon=2, batched=True, num_worlds=2,
                                      trace_worlds=(0, 1), **density)
            network = RecurrentPolicy(config.appearance_dim)
            collector = BatchedRolloutCollector(config, network, seeds=[11, 22])
            collector.world.state.life[0] = 1
            traces = []
            collector.trace_recorder = BatchedTraceRecorder(
                config, lambda kind, **values: traces.append(values))
            rollout = collector.collect(10)
            self.assertEqual(rollout.terminated.tolist(), [True, False])
            self.assertEqual(rollout.horizon_completed.tolist(), [False, True])
            self.assertEqual(encounter_count(rollout, True), 0)
            self.assertEqual(survival_loss_components(rollout).loss.item(), 0)
            self.assertEqual([(t['world'], t['step']) for t in traces], [(0, 0), (1, 0), (1, 1)])
            for trace in traces:
                self.assertEqual(trace['participants'], [-1, -1])
                self.assertEqual(trace['pairs'], [])
                self.assertEqual(trace['decisions'], [])
                self.assertEqual(trace['rewards'], [1] * 5)
