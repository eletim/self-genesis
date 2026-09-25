import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from self_genesis.config import ExperimentConfig, load_config
from self_genesis.experiment import initialize, resolve_device


class FoundationTests(unittest.TestCase):
    def test_config_file_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditions.toml"
            path.write_text('seed = 12\nnum_agents = 3\ninitial_points = 0\n')
            config = load_config(path, seed=42, device="cpu")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.num_agents, 3)
        self.assertEqual(config.initial_points, 0)

    def test_invalid_conditions(self):
        for values in (
            {"num_agents": 1}, {"appearance_dim": 0}, {"initial_life": 0},
            {"initial_points": -1}, {"seed": -1}, {"seed": 2**63},
            {"seed": True}, {"initial_life": 1.5}, {"device": "tpu"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ExperimentConfig(**values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text("typo = 1\n")
            with self.assertRaisesRegex(ValueError, "Unknown configuration keys"):
                load_config(path)

    def test_seed_and_independent_agent_state(self):
        config = ExperimentConfig(num_agents=3, appearance_dim=5)
        first = initialize(config)
        random_sample, torch_sample = random.random(), torch.rand(2)
        second = initialize(config)
        self.assertEqual(random_sample, random.random())
        self.assertTrue(torch.equal(torch_sample, torch.rand(2)))
        self.assertTrue(torch.equal(first.appearance, second.appearance))
        self.assertEqual(first.appearance.shape, (3, 5))
        self.assertFalse(torch.equal(first.appearance[0], first.appearance[1]))
        self.assertFalse(torch.equal(first.appearance, initialize(
            ExperimentConfig(seed=1, num_agents=3, appearance_dim=5)
        ).appearance))
        first.life[0] = 0
        first.points[0] = 0
        self.assertEqual(first.life.tolist(), [0, 10, 10])
        self.assertEqual(first.points.tolist(), [0, 3, 3])
        self.assertEqual(second.life.tolist(), [10, 10, 10])
        self.assertEqual(first.life.device.type, "cpu")

    def test_device_selection(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto").type, "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA was requested"):
                resolve_device("cuda")
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("auto").type, "cuda")
            self.assertEqual(resolve_device("cpu").type, "cpu")
            self.assertEqual(resolve_device("cuda").type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable")
    def test_cuda_initialization(self):
        cpu = initialize(ExperimentConfig(seed=9))
        cuda = initialize(ExperimentConfig(seed=9, device="cuda"))
        for name in ("life", "points", "appearance"):
            self.assertEqual(getattr(cuda, name).device.type, "cuda")
            self.assertTrue(torch.equal(getattr(cpu, name), getattr(cuda, name).cpu()))

    def test_cli(self):
        command = [sys.executable, "-m", "self_genesis", "--config",
                   "configs/default.toml", "--seed", "7", "--device", "cpu"]
        first = subprocess.run(command, check=True, capture_output=True, text=True)
        second = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(first.stdout, second.stdout)
        result = json.loads(first.stdout)
        self.assertEqual(result["config"]["seed"], 7)
        self.assertEqual(result["resolved_device"], "cpu")
        self.assertEqual(len(result["appearance"]), 4)
        invalid = subprocess.run(command + ["--seed", "-1"], capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("seed must be", invalid.stderr)
        self.assertNotIn("Traceback", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
