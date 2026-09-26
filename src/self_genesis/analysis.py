"""Relationship history derived from observations, never policy inputs."""

class RelationshipAnalysis:
    """Episode-local, directed histories keyed by observer and partner Appearance."""

    def __init__(self, start):
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
            key = (agent, tuple(appearance))
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
