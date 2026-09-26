"""Summarize complete CLI training runs using only the Python standard library."""

import argparse
import json
from pathlib import Path
import sys


# Keep this standard-library script runnable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_genesis.analysis import RelationshipAnalysis


def analyze(path):
    """Yield one row per completed update; reject incomplete/truncated runs."""
    start = summary = None
    episodes = 0
    with open(path, encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record['schema_version'] != 1:
                raise ValueError('Unsupported observation schema')
            kind = record['type']
            if kind == 'episode_start':
                if start is not None:
                    raise ValueError('Episode has no completed training update')
                start = record
                relationships = RelationshipAnalysis(start)
            elif start is None or record['episode'] != start['episode']:
                raise ValueError('Record has no matching episode start')
            elif kind == 'step':
                relationships.record_step(record)
            elif kind == 'summary':
                summary = record
            elif kind == 'training':
                if (summary is None or summary['truncated']
                        or not (summary['terminated'] or summary.get('horizon_completed', False))):
                    raise ValueError('Training requires a complete episode summary')
                yield {
                    'episode': record['episode'], 'device': start['resolved_device'],
                    'loss': record['loss'], 'survival_returns': record['survival_returns'],
                    'mean_survival_time': summary['mean_survival_time'],
                    'deaths': summary['deaths'], 'action_counts': summary['action_counts'],
                    'action_ratios': summary['action_ratios'],
                    'token_counts': summary['token_counts'],
                    'final_life': summary['life'], 'final_points': summary['points'],
                    'relationship_actions': relationships.rows,
                }
                episodes += 1
                expected = start['settings']['episodes']
                start = summary = None
    if start is not None or episodes == 0 or episodes != expected:
        raise ValueError('Incomplete training run')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='JSONL from self-genesis train')
    args = parser.parse_args()
    # Validate the whole run before printing results.
    for row in list(analyze(args.path)):
        print(json.dumps(row, allow_nan=False))
