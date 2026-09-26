"""Matched evaluations of learned, fixed, and oracle policies using shared world rules."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random

import torch

from self_genesis.analysis import (RelationshipAnalysis, action_metrics,
                                   communication_metrics, evaluation_summary,
                                   relationship_metrics)
from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterProtocol, Observation
from self_genesis.experiment import resolve_device
from self_genesis.policy import AgentPolicy, PolicyState, RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import train_episode
from self_genesis.world import Action, World


INTERVENTION_SEMANTICS = {
    'appearance-shuffle': (
        'Before every world step, independently permute all initial agent Appearance vectors '
        'using random.Random(evaluation_seed + 3). Use the partner-indexed permutation only '
        'in observations, consistently across all callbacks in that encounter. Fixed points '
        'and vectors belonging to dead agents are allowed. World Appearance is unchanged.'),
    'working-memory-reset': (
        'Zero only each agent Working Memory before every world step. Retain affect, weights, '
        'and all world state. Memory evolves normally between communication and action '
        'callbacks within the encounter; affect can still carry history.'),
}
INTERVENTIONS = tuple(INTERVENTION_SEMANTICS)


class _AppearanceShuffleProtocol(EncounterProtocol):
    """Resample observed identities once per encounter, without touching the world."""

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)
        self._shuffle_rng = random.Random(kwargs['seed'] + 3)

    def step(self, policies):
        self._appearance_indices = list(range(len(policies)))
        self._shuffle_rng.shuffle(self._appearance_indices)
        return super().step(policies)

    def _observe(self, agent, partner, **kwargs):
        observation = super()._observe(agent, partner, **kwargs)
        return replace(observation, partner_appearance=self.world.state.appearance[
            self._appearance_indices[partner]].clone())


class FixedPolicy:
    """Attempt the same action at every encounter; send empty messages."""

    def __init__(self, action: Action):
        self.action = action

    def communicate(self, observation):
        return ()

    def act(self, observation):
        return self.action


@dataclass(frozen=True)
class _ProducerObservation(Observation):
    partner_generation_probability: float


class _ProducerOracleProtocol(EncounterProtocol):
    """Route privileged ability only during oracle evaluation, by partner index."""

    def _observe(self, agent, partner, **kwargs):
        observation = super()._observe(agent, partner, **kwargs)
        return _ProducerObservation(
            **vars(observation),
            partner_generation_probability=float(
                self.world.state.point_generation_probability[partner]))


class ProducerOracle:
    """Evaluation-only aid to positive producers at or above the range midpoint.

    Privileged partner ability arrives only through oracle observations, never
    through learned observations, the network, or its training collector.
    """

    def __init__(self, world: World, config: ExperimentConfig):
        midpoint = (config.point_generation_probability_min
                    + config.point_generation_probability_max) / 2
        self.threshold = world.state.point_generation_probability.new_tensor(midpoint).item()

    def communicate(self, observation):
        return ()

    def act(self, observation):
        probability = observation.partner_generation_probability
        return (Action.GIVE if probability > 0 and probability >= self.threshold
                else Action.NOTHING)


class _EvaluationRecorder:
    def __init__(self, policy, index, callbacks):
        self.policy = policy
        self.index = index
        self.callbacks = callbacks

    def communicate(self, observation):
        message = tuple(self.policy.communicate(observation))
        self.callbacks.append(dict(agent=self.index, phase="communication", message=message))
        return message

    def act(self, observation):
        action = self.policy.act(observation)
        self.callbacks.append(dict(
            agent=self.index, phase="action", choice=action.value,
            observation=dict(partner_appearance=observation.partner_appearance.tolist())))
        return action


def _budget(config):
    if config.survival_horizon is not None:
        return config.survival_horizon
    if config.point_generation_probability_max > 0:
        raise ValueError("Renewable comparison requires an explicit survival_horizon")
    return config.initial_life + config.num_agents * config.initial_points


@torch.no_grad()
def evaluate_policy(config: ExperimentConfig,
                    network: RecurrentPolicy | Action | type[ProducerOracle], *,
                    intervention: str | None = None) -> dict:
    """Fresh world and policy state for one seed; never update learned weights."""
    if intervention is not None and intervention not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention: {intervention}")
    if intervention is not None and not isinstance(network, RecurrentPolicy):
        raise ValueError("Interventions require a learned policy")
    budget = _budget(config)
    world = World(config)
    protocol_type = _ProducerOracleProtocol if network is ProducerOracle else EncounterProtocol
    if intervention == 'appearance-shuffle':
        protocol_type = _AppearanceShuffleProtocol
    protocol = protocol_type(world, seed=config.seed,
                             vocabulary_size=config.vocabulary_size,
                             max_message_length=config.max_message_length)
    policies = [(ProducerOracle(world, config) if network is ProducerOracle else
                 FixedPolicy(network) if isinstance(network, Action) else AgentPolicy(network))
                for _ in range(config.num_agents)]
    initial = dict(appearance=world.state.appearance.tolist(),
                   point_generation_probability=world.state.point_generation_probability.tolist())
    relationships = RelationshipAnalysis(initial, history_by_partner=True)
    messages = []
    returns = [0.0] * config.num_agents
    death_steps = [None] * config.num_agents
    for step in range(budget):
        if intervention == 'working-memory-reset':
            for policy in policies:
                policy.state = PolicyState(torch.zeros_like(policy.state.memory),
                                           policy.state.affect, policy.state.entities)
        callbacks = []
        result = protocol.step([_EvaluationRecorder(policy, i, callbacks)
                                for i, policy in enumerate(policies)])
        relationships.record_step(dict(
            step=step, participants=[c['agent'] for c in callbacks if c['phase'] == 'action'],
            callbacks=callbacks,
            successful_transfers=[dict(donor=a, recipient=b)
                                  for a, b in result.successful_transfers],
            generated_points=result.generated_points.tolist()))
        messages.extend(dict(step=step, agent=c['agent'], message=list(c['message']))
                        for c in callbacks if c['phase'] == 'communication')
        for i, reward in enumerate(result.reward.tolist()):
            returns[i] += reward
            if result.died[i]:
                death_steps[i] = step
            if isinstance(policies[i], AgentPolicy):
                policies[i].clear_decisions()
                if result.died[i]:
                    policies[i].reset()
        if result.done:
            break
    rows = relationships.rows
    return dict(
        seed=config.seed, intervention=intervention, config=asdict(config),
        resolved_device=str(world.state.life.device),
        initial=initial, steps=step + 1, terminated=result.done,
        horizon_completed=not result.done and config.survival_horizon is not None,
        survival_returns=returns, deaths=sum(s is not None for s in death_steps),
        lifetimes=[dict(agent=i, observed_steps=age, death_step=death_steps[i],
                        censored=death_steps[i] is None) for i, age in enumerate(returns)],
        mean_observed_lifetime=sum(returns) / len(returns),
        mean_survival_time=sum(returns) / len(returns) if result.done else None,
        final_life=world.state.life.tolist(), final_points=world.state.points.tolist(),
        **action_metrics(rows), relationship_actions=rows, history_key="actual_partner",
        relationship_metrics=relationship_metrics(rows),
        communication_messages=messages,
        communication=communication_metrics([row['message'] for row in messages],
                                            config.vocabulary_size))


def run_comparison(config: ExperimentConfig, output: Path, *, evaluation_seeds=None,
                   training_seeds=None, interventions=()) -> dict:
    """Independently train each seed and evaluate matched frozen populations."""
    _budget(config)
    seeds = [config.seed] if evaluation_seeds is None else list(evaluation_seeds)
    train_seeds = [config.seed] if training_seeds is None else list(training_seeds)
    interventions = list(interventions)
    if not seeds or not train_seeds:
        raise ValueError("At least one evaluation and training seed is required")
    if len(set(seeds)) != len(seeds) or len(set(train_seeds)) != len(train_seeds):
        raise ValueError("Seeds must be unique within each seed list")
    if len(set(interventions)) != len(interventions) or any(
            item not in INTERVENTIONS for item in interventions):
        raise ValueError("Interventions must be unique supported names")
    conditions = [replace(config, seed=seed) for seed in seeds]
    training_conditions = [replace(config, seed=seed) for seed in train_seeds]
    device = resolve_device(config.device)
    # Exclusive creation prevents a repeated command from overwriting a report.
    with Path(output).open('x', encoding='utf-8') as destination:
        training_runs, evaluations, summaries, effects = [], [], [], []
        for training_config in training_conditions:
            torch.manual_seed(training_config.seed)
            network = RecurrentPolicy(
                config.appearance_dim, vocabulary_size=config.vocabulary_size,
                max_message_length=config.max_message_length,
                memory_dim=config.memory_dim, affect_dim=config.affect_dim,
                entity_memory_dim=config.entity_memory_dim).to(device)
            collector = RolloutCollector(training_config, network)
            optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
            updates = [asdict(train_episode(collector, optimizer)) for _ in range(config.episodes)]
            training_runs.append(dict(seed=training_config.seed, updates=updates))
            network.eval()
            for condition in conditions:
                for name, policy in (("learned", network), ("always-GIVE", Action.GIVE),
                                     ("always-NOTHING", Action.NOTHING),
                                     ("producer-oracle", ProducerOracle)):
                    evaluations.append(dict(training_seed=training_config.seed, policy=name,
                                            **evaluate_policy(condition, policy)))
                    if name == 'learned':
                        baseline = evaluations[-1]
                for intervention in interventions:
                    result = dict(training_seed=training_config.seed, policy="learned",
                                  **evaluate_policy(condition, network, intervention=intervention))
                    evaluations.append(result)
                    before = evaluation_summary([baseline], config.vocabulary_size)
                    after = evaluation_summary([result], config.vocabulary_size)
                    effects.append(dict(
                        training_seed=training_config.seed, evaluation_seed=condition.seed,
                        intervention=intervention,
                        mean_observed_lifetime_delta=(after['mean_observed_lifetime']
                                                      - before['mean_observed_lifetime']),
                        censored_delta=after['censored'] - before['censored'],
                        give_ratio_delta=(None if before['action_ratios']['GIVE'] is None
                                          or after['action_ratios']['GIVE'] is None else
                                          after['action_ratios']['GIVE'] - before['action_ratios']['GIVE'])))
            for name, intervention in [(name, None) for name in (
                    'learned', 'always-GIVE', 'always-NOTHING', 'producer-oracle')] + [
                        ('learned', item) for item in interventions]:
                matched = [row for row in evaluations if row['training_seed'] == training_config.seed
                           and row['policy'] == name and row['intervention'] == intervention]
                summaries.append(dict(training_seed=training_config.seed, policy=name,
                                      intervention=intervention,
                                      **evaluation_summary(matched, config.vocabulary_size)))
        report = dict(schema_version=2, config=asdict(config),
                      training=training_runs[0]['updates'] if len(training_runs) == 1 else None,
                      training_seeds=train_seeds, training_runs=training_runs,
                      evaluation_seeds=seeds, interventions=interventions,
                      intervention_semantics={name: INTERVENTION_SEMANTICS[name]
                                              for name in interventions},
                      metric_semantics=dict(
                          give_collapse='Pooled action-callback GIVE fraction within each training '
                                        'seed, policy and intervention: <= 0.05 near-always-NOTHING; '
                                        '>= 0.95 near-always-GIVE; otherwise mixed. No callbacks: '
                                        'no_actions. Descriptive evaluation diagnostic, not a '
                                        'claim about training dynamics.',
                          survival='Mean observed lifetime includes deaths and right-censored '
                                   'survivors at the evaluation budget. Uncensored mean survival '
                                   'is null if any lifetime is censored.',
                          history='Directed actual-partner history from strictly earlier steps; '
                                  'partner indices and history are analysis-only. Prior-aid GIVE '
                                  'difference compares previously aided with encountered-but-unaided '
                                  'partners, excluding unseen partners; null if either bin is empty.',
                          communication='Counts and empirical token entropy over sent messages, '
                                        'including empty-message callbacks. These measure channel '
                                        'usage, not causal utility.',
                          effects='Intervention minus matched untreated learned evaluation. '
                                  'Sampling streams restart; trajectories may diverge after actions '
                                  'or survival differ. Interventions are evaluated separately.'),
                      evaluations=evaluations, summaries=summaries, intervention_effects=effects)
        json.dump(report, destination, allow_nan=False, sort_keys=True)
        destination.write('\n')
    return dict(output=str(Path(output).resolve()), evaluation_seeds=seeds,
                training_seeds=train_seeds, training_episodes=config.episodes * len(train_seeds),
                evaluations=len(evaluations))
