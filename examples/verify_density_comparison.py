"""Validate retained density evidence and emit compact, per-seed results."""
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import sys

from self_genesis.analysis import evaluation_summary
from self_genesis.comparison import INTERVENTIONS
from self_genesis.config import load_config
from examples.run_density_comparison import DENSITIES, EVALUATION_SEEDS, TRAINING_SEEDS


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['runner_sha256'] == hashlib.sha256(
        Path('examples/run_density_comparison.py').read_bytes()).hexdigest()
    assert len(manifest['cases']) == 4
    config = load_config(Path('configs/controlled-density.toml'))
    compact = []
    for density, case in zip(DENSITIES, manifest['cases']):
        assert case['density'] == density and case['returncode'] == 0 and 'error' not in case
        raw = gzip.decompress((root / f'pairs-{density}.json.gz').read_bytes())
        assert hashlib.sha256(raw).hexdigest() == case['report_sha256']
        report = json.loads(raw)
        expected = json.loads(json.dumps(asdict(replace(config, encounter_count=density))))
        assert report['config'] == expected
        assert report['training_seeds'] == list(TRAINING_SEEDS)
        assert report['evaluation_seeds'] == list(EVALUATION_SEEDS)
        assert report['interventions'] == list(INTERVENTIONS)
        assert len(report['training_runs']) == 6
        assert len(report['evaluations']) == 117
        assert len(report['summaries']) == 39
        assert len(report['intervention_effects']) == 72
        exposure = []
        for run in report['training_runs']:
            assert len(run['updates']) == config.episodes
            dimension = config.entity_memory_dim if run['policy'] == 'learned' else 0
            assert run['config'] == {**expected, 'seed': run['seed'],
                                     'entity_memory_dim': dimension}
            assert run['parameter_count'] == (10987 if dimension else 3051)
            schedule = [[run['seed'] + update * config.num_worlds + row
                         for row in range(config.num_worlds)] for update in range(config.episodes)]
            assert run['world_seeds'] == schedule
            assert not set(EVALUATION_SEEDS).intersection(seed for batch in schedule for seed in batch)
            steps = encounters = 0
            for update in run['updates']:
                assert len(update['steps']) == config.num_worlds
                assert all(0 < step <= config.survival_horizon for step in update['steps'])
                count = update['metrics']['encounters']
                assert 0 <= count <= density * sum(update['steps'])
                assert update['metrics']['action_callbacks'] == 2 * count
                steps += sum(update['steps'])
                encounters += count
            exposure.append(dict(seed=run['seed'], policy=run['policy'],
                                 world_steps=steps, encounters=encounters))
        indexed = {(r['training_seed'], r['seed'], r['policy'], r['intervention']): r
                   for r in report['evaluations']}
        assert len(indexed) == 117
        for row in report['evaluations']:
            assert row['initial'] == indexed[row['training_seed'], row['seed'], 'learned', None]['initial']
            if row['policy'] == 'learned-no-entity-memory' and row['intervention'] == 'entity-memory-reset':
                baseline = indexed[row['training_seed'], row['seed'], row['policy'], None]
                assert {k: v for k, v in row.items() if k != 'intervention'} == {
                    k: v for k, v in baseline.items() if k != 'intervention'}
        for summary in report['summaries']:
            rows = [r for r in report['evaluations'] if all(
                r[k] == summary[k] for k in ('training_seed', 'policy', 'intervention'))]
            assert len(rows) == 3
            # JSON roundtrip normalizes integer distribution keys.
            expected = json.loads(json.dumps(evaluation_summary(rows, config.vocabulary_size)))
            assert all(summary[k] == value for k, value in expected.items())
        for effect in report['intervention_effects']:
            key = effect['training_seed'], effect['evaluation_seed'], effect['policy']
            before = evaluation_summary([indexed[*key, None]], config.vocabulary_size)
            after = evaluation_summary([indexed[*key, effect['intervention']]], config.vocabulary_size)
            pairs = {
                'mean_observed_lifetime_delta': (before['mean_observed_lifetime'], after['mean_observed_lifetime']),
                'censored_delta': (before['censored'], after['censored']),
                'prior_aid_give_difference_delta': (before['prior_aid_give_difference'], after['prior_aid_give_difference']),
                'give_ratio_delta': (before['action_ratios']['GIVE'], after['action_ratios']['GIVE']),
                'repeat_minus_first_producer_difference_delta': (
                    before['partner_history_metrics']['repeat_minus_first_producer_difference'],
                    after['partner_history_metrics']['repeat_minus_first_producer_difference']),
            }
            for field, (a, b) in pairs.items():
                assert effect[field] == (None if a is None or b is None else b - a)
        compact.append(dict(density=density, training_exposure=exposure,
                            summaries=report['summaries'],
                            intervention_effects=report['intervention_effects']))
    return compact


if __name__ == '__main__':
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'docs/evidence/controlled-density')
    print(json.dumps(verify(root), indent=2, allow_nan=False))
