"""Deterministic temporal comparisons independent of learned policy sampling."""

import json
from pathlib import Path
import tempfile
import unittest

from examples.analyze_run import analyze


APPEARANCES = [[0.1], [0.2], [0.3]]


def step(number, agents, choices, transfers=(), generated=(0, 0, 0)):
    return dict(type='step', step=number, participants=agents,
                callbacks=[dict(agent=agent, phase='action', choice=choice,
                                observation=dict(partner_appearance=APPEARANCES[partner]))
                           for agent, partner, choice in
                           zip(agents, reversed(agents), choices)],
                successful_transfers=[dict(donor=a, recipient=b) for a, b in transfers],
                generated_points=list(generated))


def episode(steps):
    return [dict(type='episode_start', resolved_device='cpu', settings=dict(episodes=2),
                 appearance=APPEARANCES, point_generation_probability=[0.0, 0.5, 1.0]),
            *steps,
            dict(type='summary', terminated=False, horizon_completed=True, truncated=False,
                 mean_survival_time=None, deaths=0, action_counts={'GIVE': 4, 'NOTHING': 2},
                 action_ratios={'GIVE': 2 / 3, 'NOTHING': 1 / 3}, token_counts=[5],
                 life=[1, 2, 3], points=[0, 1, 2]),
            dict(type='training', loss=1.5, survival_returns=[4, 4, 4])]


class AnalysisTests(unittest.TestCase):
    def analyze_episodes(self, episodes):
        records = []
        for number, events in enumerate(episodes):
            events[0]['settings']['episodes'] = len(episodes)
            records.extend(dict(schema_version=1, episode=number, **event) for event in events)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            return list(analyze(path))

    def test_directed_prior_history_and_episode_reset(self):
        events = episode([
            # Agent 0 attempts GIVE without aid; agent 1 supplies actual aid.
            step(0, [0, 1], ['GIVE', 'GIVE'], [(1, 0)], [0, 1, 1]),
            step(1, [0, 2], ['NOTHING', 'NOTHING'], generated=[0, 0, 1]),
            # Reverse protocol order: history belongs to the observer/Appearance.
            step(2, [1, 0], ['NOTHING', 'GIVE'], [(0, 1)]),
            step(3, [0, 2], ['GIVE', 'NOTHING']),
            step(4, [0, 1], ['NOTHING', 'GIVE']),
        ])
        rows = self.analyze_episodes([events, episode([step(0, [0, 1], ['GIVE', 'NOTHING'])])])
        actions = rows[0]['relationship_actions']
        for first in actions[:4]:
            self.assertEqual(first['prior'], dict(
                encounters=0, received_give_attempts=0, received_nothing=0,
                received_aid=0, outgoing_give_attempts=0, outgoing_aid=0))
        self.assertFalse(actions[0]['successful_aid'])
        self.assertTrue(actions[1]['successful_aid'])
        self.assertEqual(actions[0]['partner_generation_probability'], 0.5)
        self.assertEqual(actions[1]['agent_generation_probability'], 0.5)
        self.assertEqual(actions[0]['partner_prior_generated_points'], 0)
        self.assertEqual(actions[2]['partner_prior_generated_points'], 1)
        self.assertEqual(actions[6]['partner_prior_generated_points'], 2)
        self.assertEqual(actions[4]['prior'], dict(
            encounters=1, received_give_attempts=1, received_nothing=0,
            received_aid=0, outgoing_give_attempts=1, outgoing_aid=1))
        self.assertEqual(actions[5]['prior'], dict(
            encounters=1, received_give_attempts=1, received_nothing=0,
            received_aid=1, outgoing_give_attempts=1, outgoing_aid=0))
        # A previously encountered non-giver is distinct from an unseen partner.
        self.assertEqual(actions[6]['prior']['encounters'], 1)
        self.assertEqual(actions[6]['prior']['received_nothing'], 1)
        self.assertEqual(actions[6]['prior']['received_give_attempts'], 0)
        self.assertEqual(actions[8]['prior'], dict(
            encounters=2, received_give_attempts=1, received_nothing=1,
            received_aid=1, outgoing_give_attempts=2, outgoing_aid=1))
        self.assertEqual(actions[8]['action'], 'NOTHING')
        self.assertEqual(rows[1]['relationship_actions'][0]['prior']['encounters'], 0)
        self.assertEqual(rows[1]['relationship_actions'][0]['partner_prior_generated_points'], 0)
        for key in ('mean_survival_time', 'deaths', 'action_counts', 'action_ratios', 'token_counts'):
            self.assertEqual(rows[0][key], events[-2][key])
        self.assertEqual(rows[0]['final_life'], [1, 2, 3])
        self.assertEqual(rows[0]['final_points'], [0, 1, 2])
        self.assertEqual(rows[0]['loss'], 1.5)
        self.assertEqual(rows[0]['survival_returns'], [4, 4, 4])

    def test_legacy_logs_do_not_claim_success_or_generation(self):
        events = episode([step(0, [0, 1], ['GIVE', 'NOTHING']),
                          step(1, [0, 1], ['NOTHING', 'GIVE'])])
        del events[0]['point_generation_probability']
        for event in events[1:3]:
            del event['generated_points'], event['successful_transfers']
        actions = self.analyze_episodes([events])[0]['relationship_actions']
        self.assertIsNone(actions[0]['successful_aid'])
        self.assertIsNone(actions[0]['partner_generation_probability'])
        self.assertIsNone(actions[2]['partner_prior_generated_points'])
        self.assertIsNone(actions[2]['prior']['received_aid'])
        self.assertIsNone(actions[2]['prior']['outgoing_aid'])
        self.assertEqual(actions[2]['prior']['outgoing_give_attempts'], 1)

    def test_empty_encounters_and_out_of_order_history(self):
        events = episode([step(0, [], [], generated=[0, 1, 0]),
                          step(1, [0, 1], ['NOTHING', 'NOTHING'])])
        actions = self.analyze_episodes([events])[0]['relationship_actions']
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0]['partner_prior_generated_points'], 1)
        for number in (0, -1):
            events[2]['step'] = number
            with self.assertRaisesRegex(ValueError, 'strictly increasing'):
                self.analyze_episodes([events])

    def test_third_party_histories_are_directed_strictly_prior_and_reset(self):
        events = episode([
            step(0, [0, 2], ['GIVE', 'GIVE'], [(2, 0)]),
            step(1, [1, 2], ['GIVE', 'NOTHING'], [(1, 2)]),
            step(2, [0, 1], ['GIVE', 'NOTHING'], [(0, 1)]),
            step(3, [1, 0], ['GIVE', 'NOTHING']),
        ])
        reports = self.analyze_episodes([events, episode([step(0, [0, 1], ['GIVE', 'GIVE'])])])
        rows = reports[0]['relationship_actions']
        self.assertTrue(all(value == 0 for value in rows[0]['prior_third_party'].values()))
        prior = rows[4]['prior_third_party']
        self.assertEqual(prior, dict(
            agent_received_give_attempts=1, agent_received_aid=1,
            agent_outgoing_give_attempts=1, agent_outgoing_aid=0,
            partner_received_give_attempts=0, partner_received_aid=0,
            partner_outgoing_give_attempts=1, partner_outgoing_aid=1))
        # Current pair's aid is excluded even on later encounters.
        self.assertEqual(rows[7]['prior_third_party'], prior)
        self.assertTrue(all(value == 0 for value in
                            reports[1]['relationship_actions'][0]['prior_third_party'].values()))
        metrics = reports[0]['partner_history_metrics']['histories']
        received = metrics['prior.received_aid']
        self.assertEqual(received['bins']['positive']['action_callbacks'], 1)
        self.assertEqual(received['bins']['zero']['action_callbacks'], 1)
        self.assertEqual(received['positive_minus_zero_give'], 1)
        self.assertEqual(metrics['prior.outgoing_aid']['positive_minus_zero_give'], -1)
        attempts = metrics['prior_third_party.agent_outgoing_give_attempts']['bins']['positive']
        aid = metrics['prior_third_party.agent_outgoing_aid']['bins']['positive']
        self.assertGreater(attempts['action_callbacks'], aid['action_callbacks'])

    def test_producer_differences_first_repeat_and_missing_bins(self):
        events = episode([
            step(0, [0, 2], ['GIVE', 'NOTHING']),
            step(1, [2, 0], ['GIVE', 'NOTHING']),
        ])
        metrics = self.analyze_episodes([events])[0]['partner_history_metrics']
        producers = metrics['partner_producers']
        self.assertEqual(producers['first']['high_minus_low_give'], 1)
        self.assertEqual(producers['repeat']['high_minus_low_give'], -1)
        self.assertEqual(metrics['repeat_minus_first_producer_difference'], -2)
        self.assertEqual(producers['first']['bins']['high']['action_callbacks'], 1)
        self.assertTrue(producers['first']['bins']['unknown']['missing_bin'])
        self.assertIsNone(producers['first']['bins']['unknown']['action_ratios']['GIVE'])
        events[0]['point_generation_probability'] = [0.5] * 3
        metrics = self.analyze_episodes([events])[0]['partner_history_metrics']
        self.assertTrue(metrics['partner_producers']['first']['bins']['high']['missing_bin'])
        self.assertIsNone(metrics['repeat_minus_first_producer_difference'])

    def test_unknown_aid_is_not_zero_and_empty_samples_are_explicit(self):
        from self_genesis.analysis import action_metrics, partner_history_metrics, relationship_metrics
        events = episode([step(0, [0, 2], ['GIVE', 'NOTHING']),
                          step(1, [0, 2], ['GIVE', 'NOTHING']),
                          step(2, [0, 1], ['GIVE', 'NOTHING'])])
        del events[0]['point_generation_probability']
        del events[1]['successful_transfers']
        report = self.analyze_episodes([events])[0]
        rows = report['relationship_actions']
        self.assertIsNone(rows[4]['prior_third_party']['agent_outgoing_aid'])
        metrics = report['partner_history_metrics']
        bins = metrics['histories']['prior.outgoing_aid']['bins']
        self.assertEqual(bins['unknown']['action_callbacks'], 2)
        self.assertEqual(bins['unknown']['successful_aid_known_samples'], 2)
        self.assertEqual(bins['unknown']['successful_aid'], 0)
        self.assertIsNone(action_metrics(rows)['successful_aid'])
        self.assertEqual(action_metrics(rows)['successful_aid_known_samples'], 4)
        self.assertTrue(bins['zero']['missing_bin'])
        self.assertEqual(relationship_metrics(rows)['previously_received_aid']['action_callbacks'], 0)
        self.assertEqual(metrics['partner_producers']['first']['bins']['unknown']['action_callbacks'], 4)
        empty = partner_history_metrics([])
        self.assertIsNone(empty['repeat_minus_first_producer_difference'])
        self.assertTrue(empty['histories']['prior.received_aid']['bins']['positive']['missing_bin'])

    def test_duplicate_appearances_do_not_merge_partner_history(self):
        events = episode([step(0, [0, 1], ['GIVE', 'NOTHING'], [(0, 1)]),
                          step(1, [0, 2], ['GIVE', 'NOTHING'])])
        for event in events[1:3]:
            for callback in event['callbacks']:
                callback['observation']['partner_appearance'] = [0.1]
        rows = self.analyze_episodes([events])[0]['relationship_actions']
        self.assertEqual(rows[2]['prior']['encounters'], 0)
        self.assertEqual(rows[2]['prior']['outgoing_aid'], 0)

    def test_exposure_counts_use_actual_partners_and_include_idle_agents(self):
        from self_genesis.analysis import RelationshipAnalysis, encounter_exposure_metrics
        analysis = RelationshipAnalysis(dict(appearance=[[0.1]] * 5))
        for number, pairs in enumerate(([[0, 1], [2, 3]], [[1, 0], [2, 3]], [[0, 2]])):
            callbacks = [dict(agent=agent, phase='action', choice='NOTHING',
                              observation=dict(partner_appearance=[number]))
                         for pair in pairs for agent in pair]
            analysis.record_step(dict(step=number, pairs=pairs, participants=[],
                                      callbacks=callbacks, successful_transfers=[],
                                      generated_points=[0] * 5))
        exposure = analysis.encounter_exposure()
        agents = exposure['per_agent']
        self.assertEqual([a['encounters'] for a in agents], [3, 2, 3, 2, 0])
        self.assertEqual([a['repeat_encounters'] for a in agents], [1, 1, 1, 1, 0])
        self.assertEqual([a['unique_partners'] for a in agents], [2, 1, 2, 1, 0])
        self.assertEqual(exposure['encounter_count_distribution'], {0: 1, 2: 2, 3: 2})
        self.assertEqual(exposure['repeat_count_distribution'], {0: 1, 1: 4})
        self.assertEqual(exposure['same_partner_count_distribution'], {0: 14, 1: 2, 2: 4})
        self.assertEqual(exposure['directed_partner_episodes'], 20)
        self.assertEqual(exposure['repeat_fraction'], 4 / 10)
        self.assertEqual([r['prior']['encounters'] for r in analysis.rows],
                         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
        empty = RelationshipAnalysis(dict(appearance=[[0.1]] * 5)).encounter_exposure()
        self.assertIsNone(empty['repeat_fraction'])
        self.assertEqual(empty['encounter_count_distribution'], {0: 5})
        # A JSON roundtrip turns histogram keys into strings. Pool episodes,
        # never combine repeated numeric agent identities into longer histories.
        pooled = encounter_exposure_metrics(json.loads(json.dumps(agents + empty['per_agent'])))
        self.assertEqual(pooled['agent_episodes'], 10)
        self.assertEqual(pooled['directed_partner_episodes'], 40)
        self.assertEqual(pooled['same_partner_count_distribution'], {0: 34, 1: 2, 2: 4})
        self.assertEqual(pooled['repeat_encounter_callbacks'], 4)

    def test_analyzer_exposure_resets_between_episodes(self):
        reports = self.analyze_episodes([
            episode([step(0, [0, 1], ['GIVE', 'NOTHING']),
                     step(1, [1, 0], ['NOTHING', 'GIVE'])]),
            episode([step(0, [], [])]),
        ])
        self.assertEqual(reports[0]['encounter_exposure']['repeat_encounter_callbacks'], 2)
        self.assertEqual(reports[1]['encounter_exposure']['encounter_count_distribution'], {0: 3})

    def test_evaluation_exposure_and_pooled_denominators(self):
        from self_genesis.analysis import evaluation_summary
        from self_genesis.comparison import evaluate_policy
        from self_genesis.config import ExperimentConfig
        from self_genesis.world import Action
        evaluations = [evaluate_policy(ExperimentConfig(
            num_agents=4, initial_life=10, survival_horizon=3,
            encounter_count=count, seed=7), Action.NOTHING) for count in (0, 2)]
        empty, dense = [e['encounter_exposure'] for e in evaluations]
        self.assertEqual(empty['encounter_count_distribution'], {0: 4})
        self.assertEqual(dense['encounter_count_distribution'], {3: 4})
        self.assertEqual(dense['encounter_callbacks'], 12)
        self.assertEqual(dense['repeat_encounter_callbacks'], sum(
            row['prior']['encounters'] > 0 for row in evaluations[1]['relationship_actions']))
        summary = evaluation_summary(evaluations, 4)
        pooled = summary['encounter_exposure']
        self.assertEqual(pooled['agent_episodes'], 8)
        self.assertEqual(pooled['directed_partner_episodes'], 24)
        self.assertEqual(pooled['encounter_count_distribution'], {0: 4, 3: 4})
        self.assertEqual(pooled['encounter_callbacks'], summary['action_callbacks'])
        self.assertEqual(summary['successful_aid_known_samples'], 12)
        self.assertEqual(summary['give_collapse'], 'near_always_NOTHING')
        self.assertEqual(summary['communication']['callbacks'], 12)
        producer_bins = summary['partner_history_metrics']['partner_producers']
        self.assertEqual(sum(b['action_callbacks'] for group in producer_bins.values()
                             for b in group['bins'].values()), 12)
