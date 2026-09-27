import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from self_genesis.comparison import evaluate_policy, run_comparison
from self_genesis.config import CAPACITY_PRESETS, ExperimentConfig, load_config
from self_genesis.policy import RecurrentPolicy
from self_genesis.training import run_training


class ThoughtConfigurationTests(unittest.TestCase):
    def test_validation_and_presets(self):
        self.assertEqual(ExperimentConfig().thought_mode, 'recurrent')
        self.assertEqual(ExperimentConfig().think_steps, 16)
        for mode in ('recurrent', 'shallow'):
            for value in (True, None, '32', 16.0, -1, 0, 15):
                with self.subTest(mode=mode, value=value):
                    with self.assertRaisesRegex(ValueError, 'think_steps'):
                        ExperimentConfig(thought_mode=mode, think_steps=value)
        for value in ('unknown', '', True, None, 16, []):
            with self.assertRaisesRegex(ValueError, 'thought_mode'):
                ExperimentConfig(thought_mode=value)
        for preset in CAPACITY_PRESETS:
            config = load_config(Path(f'configs/policy-{preset}.toml'))
            self.assertEqual((config.thought_mode, config.think_steps), ('recurrent', 16))
            for steps in (16, 32, 64):
                config = load_config(capacity_preset=preset, thought_mode='shallow',
                                     think_steps=steps)
                network = RecurrentPolicy.from_preset(2, preset, thought_mode=config.thought_mode,
                                                       think_steps=config.think_steps)
                self.assertEqual((network.thought_mode, network.think_steps), ('shallow', steps))
                self.assertEqual(network.memory_dim, config.memory_dim)

    def test_training_and_comparison_initialization_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            for batched in (False, True):
                for mode in ('recurrent', 'shallow'):
                    counts = []
                    for steps in (16, 32, 64):
                        with self.subTest(batched=batched, mode=mode, steps=steps):
                            config = ExperimentConfig(
                                batched=batched, num_worlds=1, num_agents=2, episodes=1,
                                initial_life=1, initial_points=0, max_message_length=0,
                                memory_dim=4, affect_dim=2, entity_memory_dim=2,
                                thought_mode=mode, think_steps=steps)
                            path = Path(directory) / f'{batched}-{mode}-{steps}'
                            with patch('self_genesis.training.RecurrentPolicy',
                                       wraps=RecurrentPolicy) as constructor:
                                result = run_training(config, path.with_suffix('.jsonl'))
                            self.assertEqual(constructor.call_args.kwargs['thought_mode'], mode)
                            self.assertEqual(constructor.call_args.kwargs['think_steps'], steps)
                            expected = RecurrentPolicy(config.appearance_dim,
                                max_message_length=0, memory_dim=4, affect_dim=2,
                                entity_memory_dim=2, thought_mode=mode, think_steps=steps)
                            count = sum(p.numel() for p in expected.parameters())
                            self.assertEqual(result['parameter_count'], count)
                            counts.append(count)
                            first = json.loads(path.with_suffix('.jsonl').read_text().splitlines()[0])
                            settings = first['config' if batched else 'policy_settings']
                            self.assertEqual((settings['thought_mode'], settings['think_steps']),
                                             (mode, steps))
                            self.assertEqual(first['parameter_count'], count)
                            # Capture actual policies used for evaluation, preserving isinstance.
                            with patch('self_genesis.comparison.evaluate_policy',
                                       wraps=evaluate_policy) as evaluate:
                                run_comparison(config, path.with_suffix('.json'),
                                                        evaluation_seeds=[100])
                            for call in evaluate.call_args_list:
                                network = call.args[1]
                                if isinstance(network, RecurrentPolicy):
                                    self.assertEqual((network.thought_mode, network.think_steps),
                                                     (mode, steps))
                            report = json.loads(path.with_suffix('.json').read_text())
                            for run in report['training_runs']:
                                self.assertEqual(run['config']['thought_mode'], mode)
                                self.assertEqual(run['config']['think_steps'], steps)
                                network = RecurrentPolicy(config.appearance_dim,
                                    max_message_length=0, memory_dim=4, affect_dim=2,
                                    entity_memory_dim=run['config']['entity_memory_dim'],
                                    thought_mode=mode, think_steps=steps)
                                self.assertEqual(run['parameter_count'],
                                                 sum(p.numel() for p in network.parameters()))
                    self.assertEqual(len(set(counts)), 1)

    def test_cli_overrides_and_invalid_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = path / 'config.toml'
            config.write_text('thought_mode = "shallow"\nthink_steps = 32\n'
                              'episodes = 1\ninitial_life = 1\ninitial_points = 0\n'
                              'num_agents = 2\nmax_message_length = 0\n')
            for command in ('init', 'train', 'compare'):
                args = [sys.executable, '-m', 'self_genesis', command, '--config', str(config),
                        '--capacity-preset', 'small', '--thought-mode', 'recurrent',
                        '--think-steps', '64']
                output = path / f'{command}.json'
                if command != 'init':
                    args += ['--output', str(output)]
                completed = subprocess.run(args, check=True, capture_output=True, text=True)
                result = json.loads(output.read_text() if command == 'compare'
                                    else completed.stdout)
                self.assertEqual(result['config']['thought_mode'], 'recurrent')
                self.assertEqual(result['config']['think_steps'], 64)
                if output.exists():
                    output.unlink()
                failed = subprocess.run(args + ['--think-steps', '15'],
                                        capture_output=True, text=True)
                self.assertEqual(failed.returncode, 2)
                self.assertIn('think_steps must be an integer >= 16', failed.stderr)
                self.assertFalse(output.exists())
