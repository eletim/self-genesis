"""Matched evaluations of learned, fixed, and oracle policies using shared world rules."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import torch

from self_genesis.analysis import RelationshipAnalysis
from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterProtocol, Observation
from self_genesis.experiment import resolve_device
from self_genesis.policy import AgentPolicy, RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import train_episode
from self_genesis.world import Action, World


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


class _ActionRecorder:
    def __init__(self, policy, index, callbacks):
        self.policy = policy
        self.index = index
        self.callbacks = callbacks

    def communicate(self, observation):
        return self.policy.communicate(observation)

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


def _action_metrics(rows):
    counts = {action.value: sum(row['action'] == action.value for row in rows)
              for action in Action}
    return dict(action_counts=counts, action_callbacks=len(rows),
                action_ratios={key: count / len(rows) if rows else None
                               for key, count in counts.items()},
                successful_aid=sum(row['successful_aid'] for row in rows))


@torch.no_grad()
def evaluate_policy(config: ExperimentConfig,
                    network: RecurrentPolicy | Action | type[ProducerOracle]) -> dict:
    """Fresh world and policy state for one seed; never update learned weights."""
    budget = _budget(config)
    world = World(config)
    protocol_type = _ProducerOracleProtocol if network is ProducerOracle else EncounterProtocol
    protocol = protocol_type(world, seed=config.seed,
                             vocabulary_size=config.vocabulary_size,
                             max_message_length=config.max_message_length)
    policies = [(ProducerOracle(world, config) if network is ProducerOracle else
                 FixedPolicy(network) if isinstance(network, Action) else AgentPolicy(network))
                for _ in range(config.num_agents)]
    initial = dict(appearance=world.state.appearance.tolist(),
                   point_generation_probability=world.state.point_generation_probability.tolist())
    relationships = RelationshipAnalysis(initial)
    returns = [0.0] * config.num_agents
    death_steps = [None] * config.num_agents
    for step in range(budget):
        callbacks = []
        result = protocol.step([_ActionRecorder(policy, i, callbacks)
                                for i, policy in enumerate(policies)])
        relationships.record_step(dict(
            step=step, participants=[c['agent'] for c in callbacks], callbacks=callbacks,
            successful_transfers=[dict(donor=a, recipient=b)
                                  for a, b in result.successful_transfers],
            generated_points=result.generated_points.tolist()))
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
    # Conditions use only previous encounters; current aid never enters its own bin.
    conditions = {
        "unseen_partner": [r for r in rows if r['prior']['encounters'] == 0],
        "previously_received_aid": [r for r in rows if r['prior']['received_aid'] > 0],
        "encountered_without_received_aid": [r for r in rows
                                             if r['prior']['encounters'] > 0
                                             and r['prior']['received_aid'] == 0],
    }
    return dict(
        seed=config.seed, config=asdict(config), resolved_device=str(world.state.life.device),
        initial=initial, steps=step + 1, terminated=result.done,
        horizon_completed=not result.done and config.survival_horizon is not None,
        survival_returns=returns, deaths=sum(s is not None for s in death_steps),
        lifetimes=[dict(agent=i, observed_steps=age, death_step=death_steps[i],
                        censored=death_steps[i] is None) for i, age in enumerate(returns)],
        mean_observed_lifetime=sum(returns) / len(returns),
        mean_survival_time=sum(returns) / len(returns) if result.done else None,
        final_life=world.state.life.tolist(), final_points=world.state.points.tolist(),
        **_action_metrics(rows), relationship_actions=rows,
        relationship_metrics={key: _action_metrics(items) for key, items in conditions.items()})


def run_comparison(config: ExperimentConfig, output: Path, *, evaluation_seeds=None) -> dict:
    """Train once, then evaluate four policies separately for each matched seed."""
    _budget(config)
    seeds = [config.seed] if evaluation_seeds is None else list(evaluation_seeds)
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    conditions = [replace(config, seed=seed) for seed in seeds]
    device = resolve_device(config.device)
    # Exclusive creation prevents a repeated command from overwriting a report.
    with Path(output).open('x', encoding='utf-8') as destination:
        torch.manual_seed(config.seed)
        network = RecurrentPolicy(
            config.appearance_dim, vocabulary_size=config.vocabulary_size,
            max_message_length=config.max_message_length,
            memory_dim=config.memory_dim, affect_dim=config.affect_dim).to(device)
        collector = RolloutCollector(config, network)
        optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
        updates = [asdict(train_episode(collector, optimizer)) for _ in range(config.episodes)]
        network.eval()
        evaluations = []
        for condition in conditions:
            for name, policy in (("learned", network), ("always-GIVE", Action.GIVE),
                                 ("always-NOTHING", Action.NOTHING),
                                 ("producer-oracle", ProducerOracle)):
                evaluations.append(dict(policy=name, **evaluate_policy(condition, policy)))
        report = dict(schema_version=1, config=asdict(config), training=updates,
                      evaluation_seeds=seeds, evaluations=evaluations)
        json.dump(report, destination, allow_nan=False, sort_keys=True)
        destination.write('\n')
    return dict(output=str(Path(output).resolve()), evaluation_seeds=seeds,
                training_episodes=config.episodes, evaluations=len(evaluations))
