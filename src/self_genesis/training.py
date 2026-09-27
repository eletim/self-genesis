"""Complete-episode Actor-Critic or REINFORCE using only survival rewards."""

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os

import torch

from self_genesis.batched_rollout import BatchedRollout, BatchedRolloutCollector
from self_genesis.config import ExperimentConfig
from self_genesis.experiment import resolve_device
from self_genesis.observation import BatchedTraceRecorder, RunRecorder
from self_genesis.training_metrics import (rollout_metrics, UpdateMeasurement,
                                           measurement_metadata)
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import Rollout, RolloutCollector
from self_genesis.world import Action


@dataclass(frozen=True)
class SurvivalLoss:
    loss: torch.Tensor
    actor_loss: torch.Tensor
    value_loss: torch.Tensor
    action_entropy: torch.Tensor
    message_entropy: torch.Tensor


def survival_loss_components(
        rollout: Rollout | BatchedRollout, *, training_method: str = "actor_critic",
        value_loss_coefficient: float = 0.5,
        action_entropy_coefficient: float = 0.01,
        message_entropy_coefficient: float = 0.01) -> SurvivalLoss:
    """Sum decision losses, averaged over agents, with undiscounted returns.

    Each message (joint token log probability) and action receives only its
    owner's reward-to-go. Unselected and lone-survivor steps still contribute
    rewards. Recurrent graphs remain intact; discrete choices use score-function
    gradients. Actor-Critic uses detached value advantages and regresses values
    to survival returns; action/message entropy bonuses regularize only the loss.
    Legacy REINFORCE uses returns directly and records zero for unused components.
    Require extinction or completion of the explicit finite survival objective.
    Interrupted collection is not a completed objective and cannot be trained.
    """
    if training_method not in ("actor_critic", "reinforce"):
        raise ValueError("training_method must be actor_critic or reinforce")
    if isinstance(rollout, BatchedRollout):
        return _batched_survival_loss(
            rollout, training_method, value_loss_coefficient,
            action_entropy_coefficient, message_entropy_coefficient)
    if (not (rollout.terminated or rollout.horizon_completed) or rollout.truncated
            or any(items and items[0].step != 0 for items in rollout.experiences)):
        raise ValueError("Survival loss requires a complete episode from step zero")
    actor_terms, value_terms, action_entropies, message_entropies = [], [], [], []
    for experiences in rollout.experiences:
        reward_to_go = 0.0
        for experience in reversed(experiences):
            reward_to_go += experience.reward
            for decision in experience.decisions:
                if decision.log_prob is not None:
                    if training_method == "reinforce":
                        actor_terms.append(-decision.log_prob * reward_to_go)
                        continue
                    if decision.value is None or decision.entropy is None:
                        raise ValueError("Sampled decisions require value and entropy")
                    advantage = reward_to_go - decision.value
                    actor_terms.append(-decision.log_prob * advantage.detach())
                    value_terms.append(advantage.square())
                    entropies = (action_entropies if isinstance(decision.choice, Action)
                                 else message_entropies)
                    entropies.append(decision.entropy)
    if not actor_terms:
        if not any(rollout.experiences):
            raise ValueError("Episode has no sampled policy decisions")
        zero = torch.zeros(())
        return SurvivalLoss(zero, zero, zero, zero, zero)
    actor_loss = torch.stack(actor_terms).sum() / len(rollout.experiences)

    def average(terms):
        return (torch.stack(terms).sum() / len(rollout.experiences) if terms
                else actor_loss.new_zeros(()))

    value_loss = average(value_terms)
    action_entropy = average(action_entropies)
    message_entropy = average(message_entropies)
    loss = (actor_loss + value_loss_coefficient * value_loss
            - action_entropy_coefficient * action_entropy
            - message_entropy_coefficient * message_entropy
            if training_method == "actor_critic" else actor_loss)
    return SurvivalLoss(loss, actor_loss, value_loss, action_entropy, message_entropy)


def _batched_survival_loss(rollout, method, value_coefficient,
                          action_coefficient, message_coefficient):
    """Average complete world objectives, each normalized by starting agents."""
    if bool(rollout.truncated.any()) or bool((rollout.start_steps != 0).any()):
        raise ValueError("Survival loss requires complete episodes from step zero")
    if not rollout.experiences:
        raise ValueError("Episode has no sampled policy decisions")
    returns = torch.zeros_like(rollout.experiences[0].reward, dtype=torch.float32)
    actor_terms, value_terms, action_terms, message_terms = [], [], [], []
    for experience in reversed(rollout.experiences):
        returns = returns + experience.reward.float()
        for decision in experience.decisions:
            if decision.log_prob is None or not bool(decision.active.any()):
                continue
            log_prob = decision.log_prob[decision.active].float()
            target = returns[decision.active]
            if method == "reinforce":
                actor_terms.append((-log_prob * target).sum())
                continue
            if decision.value is None or decision.entropy is None:
                raise ValueError("Sampled decisions require value and entropy")
            advantage = target - decision.value[decision.active].float()
            actor_terms.append((-log_prob * advantage.detach()).sum())
            value_terms.append(advantage.square().sum())
            terms = message_terms if decision.communicating else action_terms
            terms.append(decision.entropy[decision.active].float().sum())
    if not actor_terms:
        zero = returns.new_zeros(())
        return SurvivalLoss(zero, zero, zero, zero, zero)
    actor = torch.stack(actor_terms).sum() / returns.numel()

    def average(terms):
        return torch.stack(terms).sum() / returns.numel() if terms else actor.new_zeros(())

    value, action, message = map(average, (value_terms, action_terms, message_terms))
    loss = (actor + value_coefficient * value - action_coefficient * action
            - message_coefficient * message if method == "actor_critic" else actor)
    return SurvivalLoss(loss, actor, value, action, message)


def survival_policy_loss(
        rollout: Rollout | BatchedRollout, *, training_method: str = "actor_critic",
        value_loss_coefficient: float = 0.5,
        action_entropy_coefficient: float = 0.01,
        message_entropy_coefficient: float = 0.01) -> torch.Tensor:
    """Return the total survival loss; REINFORCE uses no baseline or bonuses."""
    return survival_loss_components(
        rollout, training_method=training_method,
        value_loss_coefficient=value_loss_coefficient,
        action_entropy_coefficient=action_entropy_coefficient,
        message_entropy_coefficient=message_entropy_coefficient).loss


@dataclass(frozen=True)
class TrainingResult:
    loss: float
    steps: int
    survival_returns: tuple[float, ...]
    terminated: bool
    horizon_completed: bool
    training_method: str
    actor_loss: float
    value_loss: float
    action_entropy: float
    message_entropy: float
    metrics: dict


def train_episode(collector: RolloutCollector,
                  optimizer: torch.optim.Optimizer) -> TrainingResult:
    """Reset, collect a complete survival objective, and update shared weights.

    Pass an optimizer for collector.network. Reset preserves sampling streams
    across episodes. Results contain no retained autograd graph.
    """
    config = collector.config
    budget = _training_budget(config)
    collector.reset()
    rollout = collector.collect(budget)
    components = survival_loss_components(
        rollout, training_method=config.training_method,
        value_loss_coefficient=config.value_loss_coefficient,
        action_entropy_coefficient=config.action_entropy_coefficient,
        message_entropy_coefficient=config.message_entropy_coefficient)
    optimizer.zero_grad(set_to_none=True)
    if components.loss.requires_grad:
        components.loss.backward()
    collector.detach()
    metrics = rollout_metrics(rollout, collector.network)
    if components.loss.requires_grad:
        optimizer.step()
    result = TrainingResult(
        components.loss.item(), rollout.steps,
        tuple(sum(item.reward for item in items) for items in rollout.experiences),
        rollout.terminated, rollout.horizon_completed, config.training_method,
        components.actor_loss.item(), components.value_loss.item(),
        components.action_entropy.item(), components.message_entropy.item(), metrics)

    if collector.recorder is not None:
        collector.recorder.record_training(result, optimizer)
    return result


@dataclass(frozen=True)
class BatchedTrainingResult:
    loss: float
    steps: tuple[int, ...]
    survival_returns: tuple[tuple[float, ...], ...]
    terminated: tuple[bool, ...]
    horizon_completed: tuple[bool, ...]
    training_method: str
    actor_loss: float
    value_loss: float
    action_entropy: float
    message_entropy: float
    metrics: dict


def train_batch(collector: BatchedRolloutCollector,
                optimizer: torch.optim.Optimizer, *,
                seeds: Sequence[int]) -> BatchedTrainingResult:
    """Reset every world, finish all episodes, then update shared weights once.

    Supply one reset seed per world and an optimizer for collector.network.
    Encounter sampling streams continue across updates. Completed worlds freeze
    until the entire batch finishes; no episode spans an optimizer update.
    Losses average per-episode agent-normalized objectives. Returned statistics
    are graph-free, with steps/endings by world and returns by world then agent.
    """
    config = collector.config
    budget = _training_budget(config)
    device = next(collector.network.parameters()).device
    _validate_mixed_precision(config, device)
    seeds = tuple(seeds)
    if (len(seeds) != collector.world.steps.numel()
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)):
        raise ValueError("Provide one integer seed in [0, 2**63) per world")
    for world, seed in enumerate(seeds):
        collector.reset(world, seed=seed)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=config.mixed_precision == "bf16"):
        rollout = collector.collect(budget)
    components = survival_loss_components(
        rollout, training_method=config.training_method,
        value_loss_coefficient=config.value_loss_coefficient,
        action_entropy_coefficient=config.action_entropy_coefficient,
        message_entropy_coefficient=config.message_entropy_coefficient)
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(components.loss)):
        collector.detach()
        raise ValueError("Non-finite batched training loss; optimizer update skipped")
    if components.loss.requires_grad:
        components.loss.backward()
    collector.detach()
    gradients = [p.grad for p in collector.network.parameters() if p.grad is not None]
    if gradients and not bool(torch.stack([torch.isfinite(grad).all() for grad in gradients]).all()):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError("Non-finite batched training gradients; optimizer update skipped")
    metrics = rollout_metrics(rollout, collector.network)
    if components.loss.requires_grad:
        optimizer.step()
    returns = torch.stack([item.reward for item in rollout.experiences]).float().sum(0)
    return BatchedTrainingResult(
        components.loss.item(), tuple(rollout.end_steps.tolist()),
        tuple(tuple(row) for row in returns.tolist()),
        tuple(rollout.terminated.tolist()), tuple(rollout.horizon_completed.tolist()),
        config.training_method, components.actor_loss.item(), components.value_loss.item(),
        components.action_entropy.item(), components.message_entropy.item(), metrics)


def _validate_mixed_precision(config: ExperimentConfig, device: torch.device) -> None:
    if config.mixed_precision == "bf16":
        if device.type != "cuda":
            raise ValueError("BF16 mixed_precision requires a CUDA device")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 mixed_precision is not supported on this CUDA device")


def _training_budget(config: ExperimentConfig) -> int:
    if config.survival_horizon is not None:
        return config.survival_horizon
    if config.point_generation_probability_max > 0:
        raise ValueError("Renewable training requires an explicit survival_horizon")
    # Without renewal, no agent outlives its Life plus all initial Points.
    return config.initial_life + config.num_agents * config.initial_points


def run_training(config: ExperimentConfig, output: Path) -> dict:
    """Apply requested reproducibility controls and restore backend settings."""
    with reproducible_execution(config):
        return _run_training(config, output)


@contextmanager
def reproducible_execution(config: ExperimentConfig):
    """Apply shared training/comparison backend controls for one run."""
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if config.deterministic and resolve_device(config.device).type == "cuda":
            workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if workspace not in (":4096:8", ":16:8"):
                if workspace is not None or torch.cuda.is_initialized():
                    raise ValueError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                                     "before CUDA initialization; restart with this environment setting")
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(config.deterministic)
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _run_training(config: ExperimentConfig, output: Path) -> dict:
    """Run a fixed number of complete updates and persist observations to JSONL."""
    _training_budget(config)  # Validate before creating an output file.
    device = resolve_device(config.device)
    _validate_mixed_precision(config, device)
    torch.manual_seed(config.seed)
    network = RecurrentPolicy(
        config.appearance_dim, vocabulary_size=config.vocabulary_size,
        max_message_length=config.max_message_length,
        memory_dim=config.memory_dim, affect_dim=config.affect_dim,
        entity_memory_dim=config.entity_memory_dim,
        thought_mode=config.thought_mode, think_steps=config.think_steps).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    if config.batched:
        return _run_batched_training(config, output, network, optimizer)
    with RunRecorder(output, training=True) as recorder:
        collector = RolloutCollector(config, network)
        # train_episode resets before collecting; do not record the unused
        # construction-time episode as part of this training run.
        collector.recorder = recorder
        total_steps = 0
        for _ in range(config.episodes):
            measurement = UpdateMeasurement(device)
            result = train_episode(collector, optimizer)
            total_steps += result.steps
            recorder._write("measurement", **measurement.finish(result.steps),
                            metadata=measurement_metadata(device))
    return {"config": asdict(config), "resolved_device": str(device),
            "output": str(Path(output).resolve()), "episodes": config.episodes,
            "total_steps": total_steps, "last_training": asdict(result),
            "parameter_count": network.parameter_count}


def _run_batched_training(config, output, network, optimizer):
    """Record one complete update per batch, with outcomes indexed by world."""
    def seeds_for(update):
        return [(config.seed + update * config.num_worlds + row) % 2**63
                for row in range(config.num_worlds)]

    total_steps = 0
    with Path(output).open("x", encoding="utf-8") as file:
        def record(kind, **values):
            file.write(json.dumps({"schema_version": 1, "type": kind, **values},
                                  allow_nan=False, sort_keys=True) + "\n")
            file.flush()

        collector = BatchedRolloutCollector(config, network, seeds=seeds_for(0))
        record("batch_run", config=asdict(config), resolved_device=str(next(network.parameters()).device),
               parameter_count=network.parameter_count, random_algorithm="philox4x32-10",
               encounter_seeds=seeds_for(0),
               measurement_metadata=measurement_metadata(next(network.parameters()).device))
        trace_recorder = BatchedTraceRecorder(config, record)
        for update in range(config.episodes):
            seeds = seeds_for(update)
            trace_recorder.update = update
            collector.trace_recorder = (trace_recorder if config.trace_worlds
                and update % config.trace_update_interval == 0 else None)
            measurement = UpdateMeasurement(next(network.parameters()).device)
            result = train_batch(collector, optimizer, seeds=seeds)
            total_steps += sum(result.steps)
            record("batch_training", update=update, seeds=seeds, **asdict(result),
                   measurement=measurement.finish(sum(result.steps)))
    return {"config": asdict(config), "resolved_device": str(next(network.parameters()).device),
            "output": str(Path(output).resolve()), "episodes": config.episodes * config.num_worlds,
            "updates": config.episodes, "total_steps": total_steps,
            "last_training": asdict(result), "parameter_count": network.parameter_count}
