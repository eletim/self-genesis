"""Entity Memory traces remain detached, analysis-only outputs."""

from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from self_genesis.analysis import RelationshipAnalysis
from self_genesis.comparison import _EvaluationRecorder, evaluate_policy
from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterExperience, Observation
from self_genesis.observation import RunRecorder, entity_memory_record
from self_genesis.policy import PolicyState, RecurrentPolicy
from self_genesis.rollout import RolloutCollector


class EntityMemoryRecordingTests(unittest.TestCase):
    def test_training_records_retrievals_callback_and_completion_writes(self):
        for dimension in (0, 5):
            with self.subTest(dimension=dimension), tempfile.TemporaryDirectory() as directory:
                config = ExperimentConfig(num_agents=2, appearance_dim=2, initial_life=3,
                                          initial_points=0, entity_memory_dim=dimension)
                network = RecurrentPolicy(2, entity_memory_dim=dimension)
                path = Path(directory) / 'run.jsonl'
                with RunRecorder(path) as recorder:
                    collector = RolloutCollector(config, network, recorder=recorder)
                    rollout = collector.collect(2)
                    records = [json.loads(line) for line in path.read_text().splitlines()]
                    saved = path.read_bytes()
                    collector.detach()
                    collector.reset()
                self.assertEqual(records[0]['policy_settings']['entity_memory_dim'], dimension)
                self.assertTrue(all(s['entity_memory'] == [] for s in records[0]['states']))
                steps = [r for r in records if r['type'] == 'step']
                for step in steps:
                    for callback in step['callbacks']:
                        decision = rollout.experiences[callback['agent']][step['step']].decisions[
                            0 if callback['phase'] == 'message' else 1]
                        trace = callback['entity_memory']
                        self.assertEqual(trace, entity_memory_record(
                            network, decision.observation.partner_appearance,
                            decision.state_before, decision.state_after))
                        self.assertEqual(trace['state_after'],
                                         callback['state_after']['entity_memory'])
                        self.assertEqual(len(trace['retrieved']), dimension)
                    for completion in step['entity_memory_completions']:
                        action = next(c for c in step['callbacks']
                                      if c['agent'] == completion['agent'] and c['phase'] == 'action')
                        self.assertEqual(completion['state_before'], action['entity_memory']['state_after'])
                        self.assertEqual(completion['state_after'],
                                         step['states'][completion['agent']]['entity_memory'])
                        if dimension:
                            self.assertNotEqual(completion['state_before'], completion['state_after'])
                self.assertEqual(steps[0]['callbacks'][0]['entity_memory']['retrieved'], [0.] * dimension)
                self.assertTrue(path.read_bytes().startswith(saved))

    def test_snapshot_has_no_tensor_alias_or_graph_and_cannot_write_back(self):
        network = RecurrentPolicy(2, entity_memory_dim=5)
        observation = Observation(3, 0, 3, 0, torch.ones(2), True, (), None)
        before = network.initial_state()
        _, after = network(observation, before, communicating=True)
        snapshot = entity_memory_record(network, observation.partner_appearance, before, after)
        original = json.dumps(snapshot)
        self.assertTrue(after.entities[0].value.requires_grad)
        snapshot['state_after'][0]['value'][0] = 999
        self.assertNotEqual(after.entities[0].value[0].item(), 999)
        with torch.no_grad():
            after.entities[0].value.add_(1)
        self.assertEqual(snapshot['state_after'][0]['value'][1:],
                         json.loads(original)['state_after'][0]['value'][1:])

    def test_evaluation_recording_preserves_outputs_under_interventions(self):
        config = ExperimentConfig(num_agents=2, appearance_dim=2, initial_life=3,
                                  initial_points=0, survival_horizon=2)
        network = RecurrentPolicy(2, entity_memory_dim=5)
        for intervention in (None, 'appearance-shuffle', 'working-memory-reset'):
            with self.subTest(intervention=intervention):
                recorded = evaluate_policy(config, network, intervention=intervention)
                with patch.object(_EvaluationRecorder, '_record_memory'):
                    plain = evaluate_policy(config, network, intervention=intervention)
                events = recorded.pop('entity_memory_events')
                self.assertEqual(plain.pop('entity_memory_events'), [])
                self.assertEqual(recorded, plain)
                self.assertEqual(len(events), 12)
                for agent in range(2):
                    local = [e for e in events if e['agent'] == agent]
                    self.assertEqual([e['phase'] for e in local],
                                     ['message', 'action', 'completion'] * 2)
                    for before, after in zip(local, local[1:]):
                        self.assertEqual(before['state_after'], after['state_before'])

    def test_hidden_metadata_and_analysis_history_cannot_reach_policy_or_writes(self):
        config = ExperimentConfig(num_agents=3, appearance_dim=2, initial_life=5,
                                  initial_points=0, survival_horizon=2)
        network = RecurrentPolicy(2, entity_memory_dim=5)
        captures = []
        original_encode = network._encode
        original_complete = network.complete_encounter

        def encode(observation, communicating):
            self.assertIs(type(observation), Observation)
            self.assertEqual(set(vars(observation)), {
                'life', 'points', 'partner_life', 'partner_points',
                'partner_appearance', 'first', 'received_message', 'partner_action'})
            result = original_encode(observation, communicating)
            captures.append(result.detach().clone())
            return result

        def complete(experience, state):
            self.assertIs(type(experience), EncounterExperience)
            self.assertEqual(set(vars(experience)), {'observation', 'action', 'gave', 'received'})
            self.assertEqual(set(vars(state)), {'memory', 'affect', 'entities'})
            return original_complete(experience, state)

        with patch.object(network, '_encode', side_effect=encode), patch.object(
                network, 'complete_encounter', side_effect=complete):
            baseline = evaluate_policy(config, network)
            expected = captures[:]
            captures.clear()
            original_record = RelationshipAnalysis.record_step

            def poisoned_history(analysis, step):
                original_record(analysis, step)
                # These are output dictionaries containing actual IDs and histories.
                for row in analysis.rows:
                    row['agent'] = 999
                    row['partner'] = 998
                    row['prior'] = {'encounters': 999, 'received_aid': 999}

            # Tiny nonzero hidden abilities cannot generate a Point for these fixed draws.
            # Thus participant-visible resources stay identical while metadata changes.
            hidden = replace(config, point_generation_probability_min=1e-12,
                             point_generation_probability_max=1e-12)
            with patch.object(RelationshipAnalysis, 'record_step', poisoned_history):
                changed = evaluate_policy(hidden, network)
        self.assertNotEqual(baseline['initial']['point_generation_probability'],
                            changed['initial']['point_generation_probability'])
        self.assertEqual(baseline['entity_memory_events'], changed['entity_memory_events'])
        self.assertEqual(len(expected), len(captures))
        for first, second in zip(expected, captures):
            self.assertTrue(torch.equal(first, second))
        self.assertEqual({f.name for f in fields(PolicyState)}, {'memory', 'affect', 'entities'})
