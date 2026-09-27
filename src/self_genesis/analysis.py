"""Evaluation metrics and relationship histories, never policy inputs."""

from collections import Counter
import math


class RelationshipAnalysis:
    """Episode-local directed histories, keyed by actual partner by default.

    Actual identities retain true history under Appearance interventions and
    collisions. Legacy callers can opt into Appearance keys. All history is
    analysis-only.
    """

    def __init__(self, start, *, history_by_partner=True):
        self.history_by_partner = history_by_partner
        self.appearance = start['appearance']
        self.probability = start.get('point_generation_probability')
        self.generated = [0] * len(self.appearance)
        self.history = {}
        self.directed_aid = {}
        self.encounter_counts = [Counter() for _ in self.appearance]
        self.producer_midpoint = (None if self.probability is None else
                                  (min(self.probability) + max(self.probability)) / 2)
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
            pairs = record.get('pairs', [record['participants']])
            partner, = [other for pair in pairs if agent in pair for other in pair if other != agent]
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
                'partner_producer_bin': (
                    'unknown' if self.probability is None else
                    'high' if self.probability[partner] > self.producer_midpoint else 'low'),
                'producer_midpoint': self.producer_midpoint,
                'prior_third_party': self._third_party_history(agent, partner),
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
        for _, _, agent, partner, action, success in pending:
            self.encounter_counts[agent][partner] += 1
            attempts, aid = self.directed_aid.get((agent, partner), (0, 0))
            self.directed_aid[agent, partner] = (
                attempts + (action == 'GIVE'),
                None if aid is None or success is None else aid + success)
        generated = record.get('generated_points')
        self.generated = [
            None if generated is None or total is None else total + generated[i]
            for i, total in enumerate(self.generated)]

    def encounter_exposure(self):
        """Episode totals by actual identity, including agents never selected."""
        partners = self.encounter_counts
        agents = []
        for agent, counts in enumerate(partners):
            distribution = Counter(counts.values())
            distribution[0] += len(partners) - 1 - len(counts)
            agents.append(dict(
                agent=agent, encounters=sum(counts.values()),
                repeat_encounters=sum(count - 1 for count in counts.values()),
                unique_partners=len(counts),
                same_partner_count_distribution=dict(sorted(distribution.items()))))
        return dict(per_agent=agents, **encounter_exposure_metrics(agents))

    def _third_party_history(self, agent, partner):
        """Prior directed aid involving either participant and anyone else."""
        result = {}
        for name, focal in (('agent', agent), ('partner', partner)):
            for direction in ('received', 'outgoing'):
                pairs = [(other, focal) if direction == 'received' else (focal, other)
                         for other in range(len(self.appearance))
                         if other not in (agent, partner)]
                histories = [self.directed_aid.get(pair, (0, 0)) for pair in pairs]
                result[f'{name}_{direction}_give_attempts'] = sum(h[0] for h in histories)
                result[f'{name}_{direction}_aid'] = (
                    None if any(h[1] is None for h in histories)
                    else sum(h[1] for h in histories))
        return result


def encounter_exposure_metrics(agents):
    """Pool agent-episode counts without joining identities across episodes.

    Same-partner bins count directed possible partner pairs, including unseen
    partners at zero. Encounters count one action callback per participant.
    """
    same_partner = Counter()
    for agent in agents:
        for count, samples in agent['same_partner_count_distribution'].items():
            same_partner[int(count)] += samples
    encounters = sum(agent['encounters'] for agent in agents)
    repeats = sum(agent['repeat_encounters'] for agent in agents)
    return dict(
        agent_episodes=len(agents), encounter_callbacks=encounters,
        repeat_encounter_callbacks=repeats,
        repeat_fraction=repeats / encounters if encounters else None,
        encounter_count_distribution=dict(sorted(Counter(a['encounters'] for a in agents).items())),
        repeat_count_distribution=dict(sorted(Counter(a['repeat_encounters'] for a in agents).items())),
        same_partner_count_distribution=dict(sorted(same_partner.items())),
        directed_partner_episodes=sum(same_partner.values()))


def action_metrics(rows):
    """Attempt frequencies over encounter action callbacks, including failed aid."""
    counts = {action: sum(row['action'] == action for row in rows)
              for action in ('GIVE', 'NOTHING')}
    return dict(action_counts=counts, action_callbacks=len(rows),
                action_ratios={key: count / len(rows) if rows else None
                               for key, count in counts.items()},
                successful_aid=(None if any(row['successful_aid'] is None for row in rows)
                                else sum(row['successful_aid'] for row in rows)),
                successful_aid_known_samples=sum(row['successful_aid'] is not None for row in rows),
                missing_bin=not rows)


def relationship_metrics(rows):
    """Descriptive conditions determined exclusively by earlier encounters."""
    conditions = {
        'unseen_partner': [r for r in rows if r['prior']['encounters'] == 0],
        'previously_received_aid': [r for r in rows if r['prior']['received_aid'] is not None
                                    and r['prior']['received_aid'] > 0],
        'encountered_without_received_aid': [r for r in rows
                                             if r['prior']['encounters'] > 0
                                             and r['prior']['received_aid'] == 0],
    }
    return {key: action_metrics(items) for key, items in conditions.items()}


def partner_history_metrics(rows):
    """Subsequent GIVE by prior aid; samples are action callbacks, not agents."""
    histories = {}
    fields = [('prior', field) for field in (
        'received_give_attempts', 'received_aid', 'outgoing_give_attempts', 'outgoing_aid')]
    fields += [('prior_third_party', f'{who}_{direction}_{kind}')
               for who in ('agent', 'partner') for direction in ('received', 'outgoing')
               for kind in ('give_attempts', 'aid')]
    for source, field in fields:
        # Direct histories compare repeat encounters only; third-party histories
        # can already exist at a first meeting of this pair.
        eligible = [r for r in rows if source != 'prior' or r['prior']['encounters'] > 0]
        bins = {'positive': [], 'zero': [], 'unknown': []}
        for row in eligible:
            value = row.get(source, {}).get(field)
            bins['unknown' if value is None else 'positive' if value > 0 else 'zero'].append(row)
        metrics = {name: action_metrics(items) for name, items in bins.items()}
        positive = metrics['positive']['action_ratios']['GIVE']
        zero = metrics['zero']['action_ratios']['GIVE']
        histories[f'{source}.{field}'] = dict(
            bins=metrics, positive_minus_zero_give=(
                None if positive is None or zero is None else positive - zero))
    producers = {}
    for encounter in ('first', 'repeat'):
        selected = [r for r in rows if (r['prior']['encounters'] == 0) == (encounter == 'first')]
        bins = {name: action_metrics([r for r in selected if r.get('partner_producer_bin', 'unknown') == name])
                for name in ('high', 'low', 'unknown')}
        high, low = (bins[name]['action_ratios']['GIVE'] for name in ('high', 'low'))
        producers[encounter] = dict(
            bins=bins, high_minus_low_give=None if high is None or low is None else high - low)
    first, repeat = (producers[name]['high_minus_low_give'] for name in ('first', 'repeat'))
    return dict(histories=histories, partner_producers=producers,
                repeat_minus_first_producer_difference=(
                    None if first is None or repeat is None else repeat - first))


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
        encounter_exposure=encounter_exposure_metrics([
            agent for evaluation in evaluations
            for agent in evaluation['encounter_exposure']['per_agent']]),
        give_collapse=('no_actions' if give is None else
                       'near_always_GIVE' if give >= 0.95 else
                       'near_always_NOTHING' if give <= 0.05 else 'mixed'),
        lifetimes=len(lifetimes), censored=censored, deaths=len(lifetimes) - censored,
        mean_observed_lifetime=observed_mean,
        mean_survival_time=None if censored else observed_mean,
        relationship_metrics=history,
        partner_history_metrics=partner_history_metrics(rows),
        prior_aid_give_difference=None if aided is None or unaided is None else aided - unaided,
        communication=communication_metrics(messages, vocabulary_size))
