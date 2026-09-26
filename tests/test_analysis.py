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
