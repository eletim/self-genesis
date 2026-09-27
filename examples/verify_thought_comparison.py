"""Recompute every retained summary and intervention effect; reject incomplete evidence."""
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

from self_genesis.analysis import evaluation_summary
from self_genesis.comparison import INTERVENTIONS, THOUGHT_CONDITIONS, THOUGHT_SAMPLE_STEPS
from self_genesis.config import load_config


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['runner_sha256'] == hashlib.sha256(
        Path('examples/run_thought_comparison.py').read_bytes()).hexdigest()
    assert manifest['training_seeds'] == [10000, 20000, 30000]
    assert manifest['evaluation_seeds'] == [100000, 100001, 100002]
    assert manifest['timeout_seconds_per_seed'] == 1200
    config = load_config(Path('configs/controlled-thought.toml'))
    expected = json.loads(json.dumps(asdict(config)))
    assert manifest['config'] == expected
    assert len(manifest['cases']) == 3
    compact = []
    for seed, case in zip(manifest['training_seeds'], manifest['cases']):
        assert case['seed'] == seed and case['returncode'] == 0 and 'error' not in case
        assert case['telemetry'] and not case['telemetry_errors']
        raw = gzip.decompress((root / f'seed-{seed}.json.gz').read_bytes())
        assert hashlib.sha256(raw).hexdigest() == case['report_sha256']
        report = json.loads(raw)
        assert report['schema_version'] == 4 and report['config'] == expected
        assert report['training_seeds'] == [seed]
        assert report['evaluation_seeds'] == manifest['evaluation_seeds']
        assert report['interventions'] == list(INTERVENTIONS)
        assert len(report['training_runs']) == 3
        assert len(report['evaluations']) == 54
        assert len(report['summaries']) == 18
        assert len(report['intervention_effects']) == 36
        exposure = []
        for run, (name, mode, depth) in zip(report['training_runs'], THOUGHT_CONDITIONS):
            assert run['policy'] == name and run['seed'] == seed
            assert run['config'] == asdict(replace(config, seed=seed, thought_mode=mode,
                                                   think_steps=depth)) | {'trace_worlds': []}
            assert len(run['updates']) == config.episodes
            assert run['parameter_count'] == (10987 if mode == 'shallow' else 11243)
            schedule = [[seed + update * config.num_worlds + row
                         for row in range(config.num_worlds)] for update in range(config.episodes)]
            assert run['world_seeds'] == schedule
            assert not set(report['evaluation_seeds']).intersection(s for batch in schedule for s in batch)
            steps = encounters = 0
            gradients = []
            for update in run['updates']:
                assert len(update['steps']) == config.num_worlds
                assert all(0 < s <= config.survival_horizon for s in update['steps'])
                count = update['metrics']['encounters']
                assert 0 <= count <= config.encounter_count * sum(update['steps'])
                assert update['metrics']['action_callbacks'] == 2 * count
                assert math.isfinite(update['loss'])
                gradients.append(update['metrics']['gradient_norm'])
                assert math.isfinite(gradients[-1])
                steps += sum(update['steps'])
                encounters += count
            exposure.append(dict(policy=name, world_steps=steps, encounters=encounters,
                                 gradient_norm_range=[min(gradients), max(gradients)]))
        indexed = {(r['seed'], r['policy'], r['intervention']): r for r in report['evaluations']}
        assert len(indexed) == 54
        dynamics = []
        for row in report['evaluations']:
            assert row['initial'] == indexed[row['seed'], 'shallow', None]['initial']
            assert row['training_seed'] == seed
            samples = row['thought_samples']
            learned = row['policy'] in {c[0] for c in THOUGHT_CONDITIONS}
            assert len(samples) == (6 if learned else 0)
            if learned:
                assert {(s['step'], s['phase']) for s in samples} == {
                    (s, p) for s in THOUGHT_SAMPLE_STEPS for p in ('communication', 'action')}
            for sample in samples:
                records = sample['dynamics']
                depth = 1 if row['policy'] == 'shallow' else row['config']['think_steps']
                assert [r['step'] for r in records] == list(range(1, depth + 1))
                for record in records:
                    assert all(math.isfinite(record[k]) for k in (
                        'thought_norm', 'change_norm', 'cosine_similarity', 'relative_change',
                        'saturation_fraction', 'value', 'value_change'))
                    assert all(math.isfinite(v) for k in ('action_logits', 'action_logit_changes')
                               for v in record[k])
                    assert record['converged'] == (record['relative_change'] <= 0.001)
                if row['seed'] == 100000 and row['intervention'] is None and sample['step'] == 0 and sample['phase'] == 'action':
                    dynamics.append(dict(policy=row['policy'], **sample))
        for summary in report['summaries']:
            rows = [r for r in report['evaluations'] if all(
                r[k] == summary[k] for k in ('training_seed', 'policy', 'intervention'))]
            assert len(rows) == 3
            recomputed = json.loads(json.dumps(evaluation_summary(rows, config.vocabulary_size)))
            assert all(summary[k] == value for k, value in recomputed.items())
        for effect in report['intervention_effects']:
            key = effect['evaluation_seed'], effect['policy']
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
        compact.append(dict(training_seed=seed, training_exposure=exposure,
                            summaries=report['summaries'], representative_thought=dynamics,
                            intervention_effects=report['intervention_effects']))
    return compact


if __name__ == '__main__':
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'docs/evidence/controlled-thought')
    print(json.dumps(verify(root), indent=2, allow_nan=False))
