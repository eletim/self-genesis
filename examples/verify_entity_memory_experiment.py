"""Check the retained matched experiment and print its reviewable summaries.

Run from the repository root; optional arguments are v0.0.5 and v0.0.6 reports
(plain JSON or gzip), in that order. Assertions deliberately fail on mismatches.
"""

import gzip
import json
from pathlib import Path
import sys

from self_genesis.analysis import evaluation_summary


def read_report(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)


def verify(old, new):
    assert old['schema_version'] == 2 and new['schema_version'] == 3
    assert old['training_seeds'] == new['training_seeds'] == [41, 42, 43]
    assert old['evaluation_seeds'] == new['evaluation_seeds'] == [101, 102, 103]
    assert {k: v for k, v in new['config'].items() if k != 'entity_memory_dim'} == old['config']
    assert new['config']['entity_memory_dim'] == 16
    assert new['config']['episodes'] == new['config']['survival_horizon'] == 100
    assert len(old['evaluations']) == 54 and len(new['evaluations']) == 117
    assert len(new['training_runs']) == 6
    for run in new['training_runs']:
        assert len(run['updates']) == 100
        assert all(0 < update['steps'] <= 100 for update in run['updates'])
        if run['policy'] == 'learned-no-entity-memory':
            reference = next(r for r in old['training_runs'] if r['seed'] == run['seed'])
            assert run['updates'] == reference['updates']

    def key(row):
        return row['training_seed'], row['seed'], row['policy'], row['intervention']

    indexed = {key(row): row for row in new['evaluations']}
    assert len(indexed) == 117
    # New history/memory diagnostics enrich rows; compare shared behavioral fields.
    fields = ('initial', 'steps', 'terminated', 'horizon_completed', 'survival_returns',
              'lifetimes', 'final_life', 'final_points', 'action_counts',
              'communication_messages')
    for row in old['evaluations']:
        seed, world, policy, intervention = key(row)
        policy = 'learned-no-entity-memory' if policy == 'learned' else policy
        matched = indexed[seed, world, policy, intervention]
        for field in fields:
            assert row[field] == matched[field], (key(row), field)
        for group, metrics in row['relationship_metrics'].items():
            assert all(matched['relationship_metrics'][group][k] == v
                       for k, v in metrics.items())
    for row in new['evaluations']:
        seed, world, policy, intervention = key(row)
        assert row['initial'] == indexed[seed, world, 'learned', None]['initial']
        if policy == 'learned-no-entity-memory' and intervention == 'entity-memory-reset':
            untreated = indexed[seed, world, policy, None]
            assert {k: v for k, v in row.items() if k != 'intervention'} == {
                k: v for k, v in untreated.items() if k != 'intervention'}
        if policy in ('always-GIVE', 'always-NOTHING', 'producer-oracle'):
            reference = indexed[41, world, policy, None]
            assert {k: v for k, v in row.items() if k != 'training_seed'} == {
                k: v for k, v in reference.items() if k != 'training_seed'}
    for summary in new['summaries']:
        rows = [r for r in new['evaluations'] if all(
            r[k] == summary[k] for k in ('training_seed', 'policy', 'intervention'))]
        assert len(rows) == 3
        expected = evaluation_summary(rows, new['config']['vocabulary_size'])
        assert all(summary[k] == value for k, value in expected.items())
    return dict(training_steps=[dict(seed=r['seed'], policy=r['policy'],
                                    steps=sum(u['steps'] for u in r['updates']))
                                for r in new['training_runs']],
                summaries=new['summaries'], intervention_effects=new['intervention_effects'])


if __name__ == '__main__':
    paths = sys.argv[1:] or ['docs/evidence/matched-learning/actor_critic.json.gz',
                            'docs/evidence/entity-memory/v006.json.gz']
    if len(paths) != 2:
        raise SystemExit('usage: verify_entity_memory_experiment.py [V005 V006]')
    print(json.dumps(verify(*(read_report(path) for path in paths)), indent=2, sort_keys=True))
