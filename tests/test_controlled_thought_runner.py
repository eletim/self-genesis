"""The bounded experiment records failures and never silently changes its budget."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


class ControlledThoughtRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The example's sibling telemetry module is also used by its direct CLI.
        with patch.object(sys, 'path', [str(Path('examples').resolve()), *sys.path]):
            spec = importlib.util.spec_from_file_location('thought_runner', 'examples/run_thought_comparison.py')
            cls.runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.runner)

    def test_manifest_precedes_workers_and_failure_does_not_skip_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            calls = []

            def worker(command, **kwargs):
                manifest = json.loads((output / 'manifest.json').read_text())
                self.assertEqual(manifest['training_seeds'], [10000, 20000, 30000])
                self.assertEqual(manifest['evaluation_seeds'], [100000, 100001, 100002])
                self.assertEqual(manifest['config']['episodes'], 50)
                self.assertEqual(manifest['config']['num_worlds'], 128)
                self.assertEqual(manifest['config']['encounter_count'], 16)
                self.assertEqual(kwargs['timeout'], 1200)
                seed = int(command[command.index('--worker') + 1])
                calls.append(seed)
                if seed == 10000:
                    raise subprocess.TimeoutExpired(command, 1200)
                if seed == 20000:
                    return subprocess.CompletedProcess(command, 1, '', 'fixture failure')
                Path(command[-1]).write_text('{"complete": true}\n')
                return subprocess.CompletedProcess(command, 0, '', '')

            telemetry = Mock(samples=[{'utilization_percent': 50}], errors=[])
            with patch.object(sys, 'argv', ['runner', '--output', str(output)]), \
                    patch.object(self.runner.torch.cuda, 'get_device_name', return_value='fixture'), \
                    patch.object(self.runner.subprocess, 'check_output', return_value='revision\n'), \
                    patch.object(self.runner.subprocess, 'run', side_effect=worker), \
                    patch.object(self.runner, 'Telemetry', return_value=telemetry):
                with self.assertRaises(SystemExit) as result:
                    self.runner.main()
                self.assertEqual(result.exception.code, 1)
                self.assertEqual(calls, [10000, 20000, 30000])
                manifest = json.loads((output / 'manifest.json').read_text())
                self.assertIn('error', manifest['cases'][0])
                self.assertIn('fixture failure', manifest['cases'][1]['error'])
                self.assertEqual(manifest['cases'][2]['report_sha256'],
                                 hashlib.sha256(b'{"complete": true}\n').hexdigest())
                self.assertTrue((output / 'seed-30000.json.gz').exists())
                self.assertFalse((output / 'seed-30000.json').exists())
                before = (output / 'manifest.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    self.runner.main()
                self.assertEqual((output / 'manifest.json').read_bytes(), before)
                self.assertEqual(len(calls), 3)

    def test_worker_uses_existing_matched_matrix(self):
        with patch.object(sys, 'argv', ['runner', '--worker', '20000', '--output', '/tmp/unused.json']), \
                patch.object(self.runner, 'run_comparison') as compare:
            self.runner.main()
        args, kwargs = compare.call_args
        self.assertEqual(args[0].survival_horizon, 32)
        self.assertEqual(kwargs, dict(training_seeds=[20000],
                                     evaluation_seeds=(100000, 100001, 100002),
                                     compare_thought=True))


if __name__ == '__main__':
    unittest.main()
