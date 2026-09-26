"""Evaluation metrics and relationship histories, never policy inputs."""

import math


class RelationshipAnalysis:
    """Episode-local directed histories, keyed by Appearance by default.

    Evaluations may key by actual partner to retain true history under Appearance
    interventions and collisions. Identity and history remain analysis-only.
    """

    def __init__(self, start, *, history_by_partner=False):
        self.history_by_partner = history_by_partner
        self.appearance = start['appearance']
        self.probability = start.get('point_generation_probability')
        self.generated = [0] * len(self.appearance)
        self.history = {}
        self.rows = []
        self.last_step = -1

    def record_step(self, record):
        if record['step'] <= self.last_step:
            raise ValueError('Analysis requires strictly increasing episode steps')
        self.last_step = record['step']
        actions = {c['agent']: c for c in record['callbacks'] if c['phase'] == 'action'}
        transfers = record.get('successful_transfers')
        transfers = (None if transfers is None else
                     {(t['donor'], t['recipient']) for t in transfers})
        pending = []
        for agent, callback in actions.items():
            partner, = [i for i in record['participants'] if i != agent]
            appearance = callback['observation']['partner_appearance']
            key = (agent, partner if self.history_by_partner else tuple(appearance))
            prior = self.history.get(key, dict(
                encounters=0, received_give_attempts=0, received_nothing=0,
                received_aid=0, outgoing_give_attempts=0, outgoing_aid=0))
            success = None if transfers is None else (agent, partner) in transfers
            self.rows.append({
                'step': record['step'], 'agent': agent, 'partner': partner,
                'partner_appearance': appearance, 'action': callback['choice'],
                'successful_aid': success,
                'agent_generation_probability': (
                    None if self.probability is None else self.probability[agent]),
                'partner_generation_probability': (
                    None if self.probability is None else self.probability[partner]),
                'agent_prior_generated_points': self.generated[agent],
                'partner_prior_generated_points': self.generated[partner],
                'prior': dict(prior),
            })
            pending.append((key, prior, agent, partner, callback['choice'], success))
        # Neither callback nor same-step aid/generation may enter prior history.
        for key, prior, agent, partner, action, success in pending:
            updated = dict(prior)
            updated['encounters'] += 1
            updated['received_give_attempts'] += actions[partner]['choice'] == 'GIVE'
            updated['received_nothing'] += actions[partner]['choice'] == 'NOTHING'
            updated['outgoing_give_attempts'] += action == 'GIVE'
            for field, aided in (
                    ('received_aid', None if transfers is None else (partner, agent) in transfers),
                    ('outgoing_aid', success)):
                updated[field] = (None if aided is None or prior[field] is None
                                  else prior[field] + aided)
            self.history[key] = updated
        generated = record.get('generated_points')
        self.generated = [
            None if generated is None or total is None else total + generated[i]
            for i, total in enumerate(self.generated)]


def action_metrics(rows):
    """Attempt frequencies over encounter action callbacks, including failed aid."""
    counts = {action: sum(row['action'] == action for row in rows)
              for action in ('GIVE', 'NOTHING')}
    return dict(action_counts=counts, action_callbacks=len(rows),
                action_ratios={key: count / len(rows) if rows else None
                               for key, count in counts.items()},
                successful_aid=sum(row['successful_aid'] for row in rows))


def relationship_metrics(rows):
    """Descriptive conditions determined exclusively by earlier encounters."""
    conditions = {
        'unseen_partner': [r for r in rows if r['prior']['encounters'] == 0],
        'previously_received_aid': [r for r in rows if r['prior']['received_aid'] > 0],
        'encountered_without_received_aid': [r for r in rows
                                             if r['prior']['encounters'] > 0
                                             and r['prior']['received_aid'] == 0],
    }
    return {key: action_metrics(items) for key, items in conditions.items()}


def communication_metrics(messages, vocabulary_size):
    """Observed channel usage; token diversity does not establish causal utility."""
    counts = [sum(message.count(token) for message in messages)
              for token in range(vocabulary_size)]
    total = sum(counts)
    return dict(callbacks=len(messages), nonempty_messages=sum(bool(m) for m in messages),
                tokens=total, token_counts=counts,
                mean_length=total / len(messages) if messages else None,
                token_entropy_bits=(-sum((n / total) * math.log2(n / total)
                                         for n in counts if n) if total else None))


def evaluation_summary(evaluations, vocabulary_size):
    """Pool evaluation samples within one training seed and treatment only."""
    rows = [row for evaluation in evaluations for row in evaluation['relationship_actions']]
    lifetimes = [row for evaluation in evaluations for row in evaluation['lifetimes']]
    messages = [row['message'] for evaluation in evaluations
                for row in evaluation['communication_messages']]
    actions = action_metrics(rows)
    give = actions['action_ratios']['GIVE']
    history = relationship_metrics(rows)
    aided = history['previously_received_aid']['action_ratios']['GIVE']
    unaided = history['encountered_without_received_aid']['action_ratios']['GIVE']
    censored = sum(row['censored'] for row in lifetimes)
    observed_mean = sum(row['observed_steps'] for row in lifetimes) / len(lifetimes)
    return dict(
        **actions,
        give_collapse=('no_actions' if give is None else
                       'near_always_GIVE' if give >= 0.95 else
                       'near_always_NOTHING' if give <= 0.05 else 'mixed'),
        lifetimes=len(lifetimes), censored=censored, deaths=len(lifetimes) - censored,
        mean_observed_lifetime=observed_mean,
        mean_survival_time=None if censored else observed_mean,
        relationship_metrics=history,
        prior_aid_give_difference=None if aided is None or unaided is None else aided - unaided,
        communication=communication_metrics(messages, vocabulary_size))
