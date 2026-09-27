"""Matched evaluations of learned, fixed, and oracle policies using shared world rules."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random

import torch

from self_genesis.analysis import (RelationshipAnalysis, action_metrics,
                                   partner_history_metrics,
                                   communication_metrics, evaluation_summary,
                                   relationship_metrics)
from self_genesis.batched_rollout import BatchedRolloutCollector
from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterProtocol, Observation
from self_genesis.experiment import resolve_device
from self_genesis.observation import entity_memory_record
from self_genesis.policy import AgentPolicy, PolicyState, RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import (train_episode, train_batch, reproducible_execution,
                                   _validate_mixed_precision)
from self_genesis.thought_diagnostics import probe_thought_steps
from self_genesis.world import Action, World


INTERVENTION_SEMANTICS = {
    'appearance-shuffle': (
        'Before every world step, independently permute all initial agent Appearance vectors '
        'using random.Random(evaluation_seed + 3). Use the partner-indexed permutation only '
        'in observations, consistently across all callbacks in that encounter. Fixed points '
        'and vectors belonging to dead agents are allowed. World Appearance is unchanged.'),
    'entity-memory-reset': (
        'Clear only each agent Entity Memory before every world step. Retain Working Memory, '
        'affect, weights and world state. Entity Memory updates normally throughout communication, '
        'action and encounter completion; other recurrent state can still carry history.'),
    'appearance-replacement': (
        'Before every world step, draw independent uniform [0, 1) Appearance vectors for all '
        'agents from an isolated CPU torch.Generator(evaluation_seed + 4). Present the '
        'partner-indexed vector consistently through communication, action and completion. '
        'Resample next step; preserve world Appearance and all policy state.'),
    'working-memory-reset': (
        'Zero only each agent Working Memory before every world step. Retain affect, weights, '
        'Entity Memory, and all world state. Memory evolves normally between communication and action '
        'callbacks within the encounter; affect can still carry history.'),
}
INTERVENTIONS = tuple(INTERVENTION_SEMANTICS)
THOUGHT_CONDITIONS = (("shallow", "shallow", 16),
                      ("recurrent-16", "recurrent", 16),
                      ("recurrent-32", "recurrent", 32))
THOUGHT_SAMPLE_STEPS = (0, 1, 2)


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


class _AppearanceReplacementProtocol(EncounterProtocol):
    """Present fresh random identities per encounter on a separate sampling stream."""

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)
        self._replacement_rng = torch.Generator(device="cpu").manual_seed(kwargs['seed'] + 4)

    def step(self, policies):
        self._appearances = torch.rand(
            self.world.state.appearance.shape, generator=self._replacement_rng
        ).to(self.world.state.appearance)
        return super().step(policies)

    def _observe(self, agent, partner, **kwargs):
        observation = super()._observe(agent, partner, **kwargs)
        return replace(observation, partner_appearance=self._appearances[partner].clone())


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
    def __init__(self, policy, index, callbacks, memory_events, thought_samples=None, step=0):
        self.policy = policy
        self.index = index
        self.callbacks = callbacks
        self.memory_events = memory_events
        self.thought_samples = thought_samples
        self.step = step

    def _probe(self, observation, phase):
        if (self.thought_samples is None or not isinstance(self.policy, AgentPolicy)
                or self.step not in THOUGHT_SAMPLE_STEPS
                or any(row['step'] == self.step and row['phase'] == phase
                       for row in self.thought_samples)):
            return
        self.thought_samples.append(dict(
            step=self.step, agent=self.index, phase=phase,
            dynamics=[asdict(row) for row in probe_thought_steps(
                self.policy.network, observation, self.policy.state,
                communicating=phase == 'communication')]))

    def _record_memory(self, observation, before, phase):
        if isinstance(self.policy, AgentPolicy):
            self.memory_events.append(dict(
                agent=self.index, phase=phase,
                **entity_memory_record(self.policy.network, observation.partner_appearance,
                                       before, self.policy.state)))

    def communicate(self, observation):
        before = self.policy.state if isinstance(self.policy, AgentPolicy) else None
        self._probe(observation, "communication")
        message = tuple(self.policy.communicate(observation))
        self._record_memory(observation, before, "message")
        self.callbacks.append(dict(agent=self.index, phase="communication", message=message))
        return message

    def complete_encounter(self, experience):
        complete = getattr(self.policy, "complete_encounter", None)
        if complete is not None:
            before = self.policy.state if isinstance(self.policy, AgentPolicy) else None
            complete(experience)
            self._record_memory(experience.observation, before, "completion")

    def act(self, observation):
        before = self.policy.state if isinstance(self.policy, AgentPolicy) else None
        self._probe(observation, "action")
        action = self.policy.act(observation)
        self._record_memory(observation, before, "action")
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
                    intervention: str | None = None, sample_thought: bool = False) -> dict:
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
    elif intervention == 'appearance-replacement':
        protocol_type = _AppearanceReplacementProtocol
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
    memory_history = []
    thought_samples = [] if sample_thought else None
    returns = [0.0] * config.num_agents
    death_steps = [None] * config.num_agents
    for step in range(budget):
        if intervention == 'working-memory-reset':
            for policy in policies:
                policy.state = PolicyState(torch.zeros_like(policy.state.memory),
                                           policy.state.affect, policy.state.entities)
        elif intervention == 'entity-memory-reset':
            for policy in policies:
                policy.state = PolicyState(policy.state.memory, policy.state.affect)
        callbacks = []
        memory_events = []
        result = protocol.step([_EvaluationRecorder(policy, i, callbacks, memory_events,
                                                   thought_samples, step)
                                for i, policy in enumerate(policies)])
        memory_history.extend(dict(step=step, **event) for event in memory_events)
        relationships.record_step(dict(
            step=step, participants=[c['agent'] for c in callbacks if c['phase'] == 'action'],
            callbacks=callbacks, pairs=protocol.last_pairs,
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
        partner_history_metrics=partner_history_metrics(rows),
        encounter_exposure=relationships.encounter_exposure(),
        entity_memory_events=memory_history,
        **({"thought_samples": thought_samples} if sample_thought else {}),
        communication_messages=messages,
        communication=communication_metrics([row['message'] for row in messages],
                                            config.vocabulary_size))


def run_comparison(config: ExperimentConfig, output: Path, *, evaluation_seeds=None,
                   training_seeds=None, interventions=(), compare_thought=False) -> dict:
    """Independently train each seed and evaluate matched frozen populations."""
    with reproducible_execution(config):
        return _run_comparison(config, output, evaluation_seeds=evaluation_seeds,
                               training_seeds=training_seeds, interventions=interventions,
                               compare_thought=compare_thought)


def _run_comparison(config, output, *, evaluation_seeds, training_seeds, interventions,
                    compare_thought):
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
    if compare_thought:
        if evaluation_seeds is None or set(seeds) & set(train_seeds):
            raise ValueError("Thought comparison requires explicit held-out evaluation_seeds")
        interventions = list(INTERVENTIONS)
    training_conditions = [replace(config, seed=seed) for seed in train_seeds]
    if config.batched:
        if evaluation_seeds is None:
            raise ValueError("Batched comparison requires explicit held-out evaluation_seeds")
        # Match train's complete world-seed schedule, including wraparound.
        span = config.episodes * config.num_worlds
        if any((seed - start) % 2**63 < span for seed in seeds for start in train_seeds):
            raise ValueError("Evaluation seeds must be held out from every batched training world")
    conditions = [replace(config, seed=seed, batched=False, mixed_precision="fp32",
                          trace_worlds=()) for seed in seeds]
    evaluation_options = {"sample_thought": True} if compare_thought else {}
    device = resolve_device(config.device)
    _validate_mixed_precision(config, device)
    # Exclusive creation prevents a repeated command from overwriting a report.
    with Path(output).open('x', encoding='utf-8') as destination:
        training_runs, evaluations, summaries, effects = [], [], [], []
        for training_config in training_conditions:
            learned = []
            # A zero dimension explicitly requests the disabled-only legacy condition.
            variants = [("learned", config.entity_memory_dim, config.thought_mode,
                         config.think_steps)]
            if config.entity_memory_dim:
                variants.append(("learned-no-entity-memory", 0, config.thought_mode,
                                 config.think_steps))
            if compare_thought:
                variants = [(name, config.entity_memory_dim, mode, steps)
                            for name, mode, steps in THOUGHT_CONDITIONS]
            for name, dimension, mode, steps in variants:
                learning_config = replace(training_config, entity_memory_dim=dimension,
                                          thought_mode=mode, think_steps=steps)
                torch.manual_seed(learning_config.seed)
                network = RecurrentPolicy(
                    config.appearance_dim, vocabulary_size=config.vocabulary_size,
                    max_message_length=config.max_message_length,
                    memory_dim=config.memory_dim, affect_dim=config.affect_dim,
                    entity_memory_dim=dimension, thought_mode=mode,
                    think_steps=steps).to(device)
                optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
                training_metadata = {}
                if config.batched:
                    world_seeds = [[(learning_config.seed + update * config.num_worlds + row)
                                    % 2**63 for row in range(config.num_worlds)]
                                   for update in range(config.episodes)]
                    collector = BatchedRolloutCollector(learning_config, network,
                                                        seeds=world_seeds[0])
                    updates = [asdict(train_batch(collector, optimizer, seeds=batch))
                               for batch in world_seeds]
                    training_metadata = dict(world_seeds=world_seeds,
                                             random_algorithm="philox4x32-10")
                else:
                    collector = RolloutCollector(learning_config, network)
                    updates = [asdict(train_episode(collector, optimizer))
                               for _ in range(config.episodes)]
                training_runs.append(dict(seed=learning_config.seed, policy=name,
                                          config=asdict(learning_config), updates=updates,
                                          parameter_count=network.parameter_count,
                                          **training_metadata))
                optimizer.zero_grad(set_to_none=True)
                network.eval()
                network.requires_grad_(False)
                learned.append((name, network))
            for condition in conditions:
                for name, policy in [*learned, ("always-GIVE", Action.GIVE),
                                     ("always-NOTHING", Action.NOTHING),
                                     ("producer-oracle", ProducerOracle)]:
                    evaluation_config = (replace(condition, entity_memory_dim=policy.entity_memory_dim,
                                                 thought_mode=policy.thought_mode,
                                                 think_steps=policy.think_steps)
                                         if isinstance(policy, RecurrentPolicy) else condition)
                    baseline = dict(training_seed=training_config.seed, policy=name,
                                    **evaluate_policy(evaluation_config, policy,
                                                      **evaluation_options))
                    evaluations.append(baseline)
                    if not isinstance(policy, RecurrentPolicy):
                        continue
                    for intervention in interventions:
                        result = dict(training_seed=training_config.seed, policy=name,
                                      **evaluate_policy(evaluation_config, policy,
                                                        intervention=intervention,
                                                        **evaluation_options))
                        evaluations.append(result)
                        before = evaluation_summary([baseline], config.vocabulary_size)
                        after = evaluation_summary([result], config.vocabulary_size)
                        effects.append(dict(
                            training_seed=training_config.seed, evaluation_seed=condition.seed,
                            policy=name, intervention=intervention,
                            mean_observed_lifetime_delta=(after['mean_observed_lifetime']
                                                          - before['mean_observed_lifetime']),
                            censored_delta=after['censored'] - before['censored'],
                            prior_aid_give_difference_delta=_difference(
                                after['prior_aid_give_difference'], before['prior_aid_give_difference']),
                            repeat_minus_first_producer_difference_delta=_difference(
                                after['partner_history_metrics']['repeat_minus_first_producer_difference'],
                                before['partner_history_metrics']['repeat_minus_first_producer_difference']),
                            give_ratio_delta=(None if before['action_ratios']['GIVE'] is None
                                              or after['action_ratios']['GIVE'] is None else
                                              after['action_ratios']['GIVE'] - before['action_ratios']['GIVE'])))
            for name, intervention in [(name, None) for name in (
                    *[name for name, _ in learned], 'always-GIVE', 'always-NOTHING',
                    'producer-oracle')] + [(name, item) for name, _ in learned
                                           for item in interventions]:
                matched = [row for row in evaluations if row['training_seed'] == training_config.seed
                           and row['policy'] == name and row['intervention'] == intervention]
                summaries.append(dict(training_seed=training_config.seed, policy=name,
                                      intervention=intervention,
                                      **evaluation_summary(matched, config.vocabulary_size)))
        report = dict(schema_version=4 if compare_thought else 3, config=asdict(config),
                      training_execution="batched" if config.batched else "sequential",
                      evaluation_execution="sequential-fp32",
                      training=training_runs[0]['updates'] if len(train_seeds) == 1 else None,
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
                          partner_history='Prior direct aid conditions use repeat encounters; '
                                          'third-party conditions exclude both participants from '
                                          'the other-agent set and use all encounters. Attempts '
                                          'and successful transfers are separate, with positive, '
                                          'zero and unknown bins. Partner producers are high above '
                                          'the episode population ability midrange, low otherwise. '
                                          'High-minus-low GIVE is reported for first and repeat '
                                          'encounters, then repeat minus first. Each bin reports '
                                          'callback counts and missing_bin; empty rates and '
                                          'unsupported differences are null.',
                          encounter_exposure='Per-agent episode encounter and repeat counts include '
                                             'zero-exposure agents. A repeat is a meeting after the '
                                             'first with the same actual partner. Count distributions '
                                             'use agent-episodes; same-partner distributions use all '
                                             'directed possible partner-episodes, including unseen '
                                             'partners at zero. Repeat fraction divides repeat action '
                                             'callbacks by all encounter action callbacks; null if empty.',
                          communication='Counts and empirical token entropy over sent messages, '
                                        'including empty-message callbacks. These measure channel '
                                        'usage, not causal utility.',
                          effects='Intervention minus matched untreated learned evaluation. '
                                  'Sampling streams restart; trajectories may diverge after actions '
                                  'or survival differ. Interventions are evaluated separately. '
                                  'Selection differences are null if either matched estimate lacks '
                                  'a required bin; they are descriptive, not causal identification.'),
                      evaluations=evaluations, summaries=summaries, intervention_effects=effects)
        if compare_thought:
            report['thought_comparison'] = dict(
                conditions=[dict(policy=name, thought_mode=mode, think_steps=steps,
                                 effective_think_steps=1 if mode == 'shallow' else steps)
                            for name, mode, steps in THOUGHT_CONDITIONS],
                matching='All configured world, capacity, learning and episode settings and '
                         'seed lists are shared; only Thought mode and depth change. Capacity '
                         'dimensions are matched, not parameter counts or compute.',
                holdout='Evaluation seeds exclude every batched training world seed (modulo '
                        '2**63), or every sequential training stream initialization seed.',
                sampling=dict(world_steps=THOUGHT_SAMPLE_STEPS,
                              selection='First communication and first action callback at each '
                                        'listed world step, per evaluation including interventions; '
                                        'absent callbacks are omitted. At most six probes. Incoming '
                                        'state is replayed without sampling or persistent writes.',
                              convergence_tolerance=1e-3),
                interpretation='Sampled Thought dynamics and communication usage do not '
                               'establish utility. Survival gains alone do not establish '
                               'partner-history dependence; retain missing bins and censoring.')
        json.dump(report, destination, allow_nan=False, sort_keys=True)
        destination.write('\n')
    return dict(output=str(Path(output).resolve()), evaluation_seeds=seeds,
                training_seeds=train_seeds, training_episodes=config.episodes * len(training_runs)
                * (config.num_worlds if config.batched else 1),
                evaluations=len(evaluations))


def _difference(after, before):
    """Retain missing-bin semantics in matched selection effects."""
    return None if after is None or before is None else after - before
