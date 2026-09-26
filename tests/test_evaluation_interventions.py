"""Evaluation interventions preserve world rules and isolate analysis from policies."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from self_genesis.analysis import (RelationshipAnalysis, communication_metrics,
                                   evaluation_summary)
from self_genesis.comparison import (INTERVENTIONS, FixedPolicy, _AppearanceShuffleProtocol,
                                     _AppearanceReplacementProtocol,
                                     evaluate_policy, run_comparison)
from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterProtocol, Observation
from self_genesis.policy import RecurrentPolicy
from self_genesis.world import Action, World


class InterventionTests(unittest.TestCase):
    def test_appearance_interventions_preserve_world_and_rng_streams(self):
        for protocol in (_AppearanceShuffleProtocol, _AppearanceReplacementProtocol):
            with self.subTest(protocol=protocol.__name__):
                self.check_appearance_protocol(protocol)

    def check_appearance_protocol(self, protocol):
        config = ExperimentConfig(num_agents=4, initial_life=20,
                                  point_generation_probability_max=1)
        ordinary, shuffled = World(config), World(config)
        baseline = EncounterProtocol(ordinary, seed=17)
        treatment = protocol(shuffled, seed=17)
        seen = []

        class Recorder(FixedPolicy):
            def communicate(self, observation):
                seen.append(observation)
                return super().communicate(observation)

            def act(self, observation):
                seen.append(observation)
                return super().act(observation)

        policies = [Recorder(Action.GIVE) for _ in range(config.num_agents)]
        changed = False
        for _ in range(8):
            seen.clear()
            expected = baseline.step(policies)
            normal_observations = list(seen)
            seen.clear()
            actual = treatment.step(policies)
            for normal, intervened in zip(normal_observations, seen, strict=True):
                self.assertIs(type(intervened), Observation)
                for key, value in vars(normal).items():
                    if key == 'partner_appearance':
                        changed |= not torch.equal(value, intervened.partner_appearance)
                    else:
                        self.assertEqual(value, getattr(intervened, key))
            # Each participant sees one consistent identity across its two callbacks.
            self.assertTrue(torch.equal(seen[0].partner_appearance, seen[2].partner_appearance))
            self.assertTrue(torch.equal(seen[1].partner_appearance, seen[3].partner_appearance))
            for key, value in vars(ordinary.state).items():
                self.assertTrue(torch.equal(value, getattr(shuffled.state, key)), key)
            self.assertEqual(expected.successful_transfers, actual.successful_transfers)
            self.assertTrue(torch.equal(expected.generated_points, actual.generated_points))
        self.assertTrue(changed)

    def test_replacement_preserves_world_sampling_and_policy_state(self):
        config = ExperimentConfig(num_agents=2, initial_life=20, survival_horizon=4)
        world = World(config)
        network = RecurrentPolicy(config.appearance_dim)
        original_step = EncounterProtocol.step
        previous_states = []
        appearances = []
        rng_states = []

        def checked_step(protocol, policies):
            for i, wrapper in enumerate(policies):
                if previous_states:
                    self.assertIs(wrapper.policy.state, previous_states[i])
            rng_states.append(torch.random.get_rng_state().clone())
            appearances.append(protocol._appearances.clone())
            self.assertTrue(torch.equal(protocol.world.state.appearance, world.state.appearance))
            result = original_step(protocol, policies)
            previous_states[:] = [w.policy.state for w in policies]
            return result

        with patch.object(EncounterProtocol, 'step', autospec=True, side_effect=checked_step):
            report = evaluate_policy(config, network, intervention='appearance-replacement')
        self.assertFalse(torch.equal(appearances[0], appearances[1]))
        observed = {}
        for event in report['entity_memory_events']:
            key = (event['step'], event['agent'])
            self.assertEqual(event['appearance'], observed.setdefault(key, event['appearance']))
            step = event['step']
            self.assertTrue(any(event['appearance'] == a.tolist() for a in appearances[step]))
        # Replacement consumes no global policy RNG before the first callback.
        World(config)
        self.assertTrue(torch.equal(rng_states[0], torch.random.get_rng_state()))

    def test_entity_reset_preserves_working_memory_affect_and_within_encounter_writes(self):
        config = ExperimentConfig(num_agents=2, initial_life=20, survival_horizon=4)
        network = RecurrentPolicy(config.appearance_dim)
        previous_states = []
        original_step = EncounterProtocol.step

        def checked_step(protocol, policies):
            for i, wrapper in enumerate(policies):
                state = wrapper.policy.state
                self.assertEqual(state.entities, ())
                if previous_states:
                    self.assertIs(state.memory, previous_states[i].memory)
                    self.assertIs(state.affect, previous_states[i].affect)
            result = original_step(protocol, policies)
            previous_states[:] = [w.policy.state for w in policies]
            self.assertTrue(all(len(s.entities) == 1 for s in previous_states))
            return result

        with patch.object(EncounterProtocol, 'step', autospec=True, side_effect=checked_step):
            report = evaluate_policy(config, network, intervention='entity-memory-reset')
        for event in report['entity_memory_events']:
            if event['phase'] == 'message':
                self.assertEqual(len(event['state_before']), 0)
            else:
                self.assertEqual(len(event['state_before']), 1)

    def test_memory_reset_preserves_affect_and_within_encounter_updates(self):
        config = ExperimentConfig(num_agents=2, initial_life=20, survival_horizon=4)
        network = RecurrentPolicy(config.appearance_dim)
        calls = []

        def inspect(module, args, kwargs):
            observation, state = args
            self.assertIs(type(observation), Observation)
            calls.append((kwargs['communicating'], state.memory.clone(), state.affect.clone()))

        handle = network.register_forward_pre_hook(inspect, with_kwargs=True)
        previous_affects = []
        previous_entities = []
        original_step = EncounterProtocol.step

        def checked_step(protocol, policies):
            for index, wrapper in enumerate(policies):
                state = wrapper.policy.state
                self.assertEqual(torch.count_nonzero(state.memory), 0)
                if previous_affects:
                    self.assertTrue(torch.equal(state.affect, previous_affects[index]))
                    self.assertIs(state.entities, previous_entities[index])
                    self.assertEqual(len(state.entities), 1)
            result = original_step(protocol, policies)
            previous_affects[:] = [wrapper.policy.state.affect.clone() for wrapper in policies]
            previous_entities[:] = [wrapper.policy.state.entities for wrapper in policies]
            return result

        with patch.object(EncounterProtocol, 'step', autospec=True, side_effect=checked_step):
            evaluate_policy(config, network, intervention='working-memory-reset')
        handle.remove()
        self.assertEqual(len(calls), 16)
        for communicating, memory, affect in calls:
            self.assertEqual(bool(torch.count_nonzero(memory)), not communicating)
        self.assertTrue(any(torch.count_nonzero(affect) for _, _, affect in calls[4:]))

    def test_interventions_repeat_preserve_weights_and_channel_bounds(self):
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        for device in devices:
            for length in (0, 2):
                config = ExperimentConfig(device=device, num_agents=3, initial_life=6,
                                          survival_horizon=4, max_message_length=length)
                network = RecurrentPolicy(config.appearance_dim, max_message_length=length).to(device)
                before = {key: value.clone() for key, value in network.state_dict().items()}
                baseline = evaluate_policy(config, network)
                for intervention in INTERVENTIONS:
                    first = evaluate_policy(config, network, intervention=intervention)
                    self.assertEqual(first, evaluate_policy(config, network, intervention=intervention))
                    self.assertEqual(first['initial'], baseline['initial'])
                    self.assertEqual(first['communication']['tokens'], first['action_callbacks'] * length)
                    self.assertEqual(first['communication']['callbacks'], first['action_callbacks'])
                self.assertEqual(baseline, evaluate_policy(config, network))
                for key, value in network.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]))
                self.assertTrue(all(parameter.grad is None for parameter in network.parameters()))

    def test_prior_history_tracks_actual_partner_despite_shuffle_and_collisions(self):
        analysis = RelationshipAnalysis(dict(appearance=[[0], [0], [0]]), history_by_partner=True)
        for step, partner in enumerate((1, 2, 1)):
            analysis.record_step(dict(
                step=step, participants=[0, partner], generated_points=[0, 0, 0],
                successful_transfers=[dict(donor=partner, recipient=0)],
                callbacks=[dict(agent=agent, phase='action', choice='GIVE',
                                observation=dict(partner_appearance=[step]))
                           for agent in (0, partner)]))
        self.assertEqual([r['prior']['encounters'] for r in analysis.rows], [0, 0, 0, 0, 1, 1])
        self.assertEqual([r['prior']['received_aid'] for r in analysis.rows], [0, 0, 0, 0, 1, 0])

    def test_summary_censoring_collapse_and_empty_communication(self):
        config = ExperimentConfig(num_agents=2, initial_life=2, initial_points=1)
        for action, label in ((Action.GIVE, 'near_always_GIVE'),
                              (Action.NOTHING, 'near_always_NOTHING')):
            report = evaluate_policy(config, action)
            summary = evaluation_summary([report], config.vocabulary_size)
            self.assertEqual(summary['give_collapse'], label)
            self.assertEqual(summary['mean_survival_time'], report['mean_survival_time'])
            self.assertEqual(summary['censored'], 0)
            self.assertEqual(summary['communication']['tokens'], 0)
            self.assertIsNone(summary['communication']['token_entropy_bits'])
        censored = evaluate_policy(replace(config, survival_horizon=1), Action.GIVE)
        summary = evaluation_summary([report, censored], config.vocabulary_size)
        self.assertEqual(summary['censored'], 2)
        self.assertEqual(summary['deaths'], 2)
        self.assertEqual(summary['mean_observed_lifetime'], 1.5)
        self.assertIsNone(summary['mean_survival_time'])
        self.assertIsNone(summary['prior_aid_give_difference'])
        empty = dict(censored, relationship_actions=[], communication_messages=[])
        self.assertEqual(evaluation_summary([empty], 4)['give_collapse'], 'no_actions')
        self.assertIsNone(communication_metrics([], 4)['mean_length'])
        self.assertEqual(communication_metrics([[0, 1], [0, 1]], 4)['token_entropy_bits'], 1)

    def test_collapse_thresholds_and_prior_aid_difference(self):
        report = evaluate_policy(ExperimentConfig(num_agents=2, initial_life=2), Action.GIVE)
        for count, expected in ((1, 'near_always_NOTHING'), (2, 'mixed'),
                                (18, 'mixed'), (19, 'near_always_GIVE')):
            rows = [dict(action='GIVE' if i < count else 'NOTHING', successful_aid=False,
                         prior=dict(encounters=0, received_aid=0)) for i in range(20)]
            summary = evaluation_summary([dict(report, relationship_actions=rows)], 4)
            self.assertEqual(summary['give_collapse'], expected)
            self.assertEqual(summary['action_callbacks'], 20)
        rows = [
            dict(action='GIVE', successful_aid=True, prior=dict(encounters=0, received_aid=0)),
            dict(action='GIVE', successful_aid=False, prior=dict(encounters=1, received_aid=1)),
            dict(action='NOTHING', successful_aid=False, prior=dict(encounters=1, received_aid=0)),
        ]
        summary = evaluation_summary([dict(report, relationship_actions=rows)], 4)
        self.assertEqual(summary['prior_aid_give_difference'], 1)
        self.assertEqual(summary['relationship_metrics']['unseen_partner']['action_callbacks'], 1)
        self.assertEqual(summary['relationship_metrics']['previously_received_aid']['successful_aid'], 0)

    def test_multi_seed_cli_is_independent_reproducible_and_matched(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = []
            command = [sys.executable, '-m', 'self_genesis', 'compare', '--num-agents', '2',
                       '--initial-life', '3', '--episodes', '1', '--survival-horizon', '2',
                       '--evaluation-seeds', '101', '102', '--interventions', *INTERVENTIONS]
            for index, seeds in enumerate((['7', '8'], ['7', '8'], ['8'])):
                output = Path(directory) / f'{index}.json'
                subprocess.run([*command, '--training-seeds', *seeds, '--output', str(output)],
                               check=True, capture_output=True, text=True)
                reports.append(json.loads(output.read_text()))
            self.assertEqual(reports[0], reports[1])
            report, single = reports[0], reports[2]
            self.assertEqual(len(report['evaluations']), 52)
            self.assertEqual(len(report['summaries']), 26)
            self.assertEqual(len(report['intervention_effects']), 32)
            self.assertEqual(report['training_runs'][2:], single['training_runs'])
            self.assertEqual([r for r in report['evaluations'] if r['training_seed'] == 8],
                             single['evaluations'])
            self.assertEqual([r for r in report['summaries'] if r['training_seed'] == 8],
                             single['summaries'])
            for effect in report['intervention_effects']:
                matched = [r for r in report['evaluations'] if r['policy'] == effect['policy']
                           and r['training_seed'] == effect['training_seed']
                           and r['seed'] == effect['evaluation_seed']]
                baseline = next(r for r in matched if r['intervention'] is None)
                changed = next(r for r in matched if r['intervention'] == effect['intervention'])
                self.assertEqual(effect['mean_observed_lifetime_delta'],
                                 changed['mean_observed_lifetime'] - baseline['mean_observed_lifetime'])

    def test_disabled_condition_is_separately_trained_with_matched_budget(self):
        config = ExperimentConfig(num_agents=2, initial_life=4, episodes=2,
                                  survival_horizon=3, entity_memory_dim=5)
        with tempfile.TemporaryDirectory() as directory:
            reports = []
            for name, settings in [('paired', config),
                                   ('disabled', replace(config, entity_memory_dim=0))]:
                output = Path(directory) / f'{name}.json'
                result = run_comparison(settings, output, training_seeds=[7],
                                        evaluation_seeds=[101], interventions=INTERVENTIONS)
                self.assertEqual(result['training_episodes'], 4 if name == 'paired' else 2)
                reports.append(json.loads(output.read_text()))
        paired, disabled = reports
        enabled_run, disabled_run = paired['training_runs']
        self.assertEqual(disabled_run['updates'], disabled['training_runs'][0]['updates'])
        self.assertEqual(enabled_run['config'], {**disabled_run['config'], 'entity_memory_dim': 5})
        self.assertEqual(len(enabled_run['updates']), len(disabled_run['updates']))
        paired_rows = [r for r in paired['evaluations'] if r['policy'] == 'learned-no-entity-memory']
        single_rows = [r for r in disabled['evaluations'] if r['policy'] == 'learned']
        self.assertEqual([{**r, 'policy': 'learned'} for r in paired_rows], single_rows)
        baseline = next(r for r in paired_rows if r['intervention'] is None)
        reset = next(r for r in paired_rows if r['intervention'] == 'entity-memory-reset')
        self.assertEqual({**reset, 'intervention': None}, baseline)
        self.assertTrue(all(r['seed'] == 101 for r in paired['evaluations']))
        self.assertTrue(all(r['initial'] == baseline['initial'] for r in paired['evaluations']))

    def test_invalid_conditions_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            for kwargs in (dict(training_seeds=[]), dict(training_seeds=[-1]),
                           dict(training_seeds=[1, 1]), dict(evaluation_seeds=[1, 1]),
                           dict(interventions=['unknown']),
                           dict(interventions=['appearance-shuffle'] * 2)):
                with self.assertRaises(ValueError):
                    run_comparison(ExperimentConfig(), output, **kwargs)
                self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            evaluate_policy(ExperimentConfig(), Action.GIVE, intervention='appearance-shuffle')
