"""Complete-episode Actor-Critic or REINFORCE using only survival rewards."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.experiment import resolve_device
from self_genesis.observation import RunRecorder
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
        rollout: Rollout, *, training_method: str = "actor_critic",
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
        raise ValueError("Episode has no sampled policy decisions")
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


def survival_policy_loss(
        rollout: Rollout, *, training_method: str = "actor_critic",
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
    components.loss.backward()
    collector.detach()
    optimizer.step()
    result = TrainingResult(
        components.loss.item(), rollout.steps,
        tuple(sum(item.reward for item in items) for items in rollout.experiences),
        rollout.terminated, rollout.horizon_completed, config.training_method,
        components.actor_loss.item(), components.value_loss.item(),
        components.action_entropy.item(), components.message_entropy.item())

    if collector.recorder is not None:
        collector.recorder.record_training(result, optimizer)
    return result


def _training_budget(config: ExperimentConfig) -> int:
    if config.survival_horizon is not None:
        return config.survival_horizon
    if config.point_generation_probability_max > 0:
        raise ValueError("Renewable training requires an explicit survival_horizon")
    # Without renewal, no agent outlives its Life plus all initial Points.
    return config.initial_life + config.num_agents * config.initial_points


def run_training(config: ExperimentConfig, output: Path) -> dict:
    """Run a fixed number of complete updates and persist observations to JSONL."""
    _training_budget(config)  # Validate before creating an output file.
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    network = RecurrentPolicy(
        config.appearance_dim, vocabulary_size=config.vocabulary_size,
        max_message_length=config.max_message_length,
        memory_dim=config.memory_dim, affect_dim=config.affect_dim,
        entity_memory_dim=config.entity_memory_dim).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    with RunRecorder(output) as recorder:
        collector = RolloutCollector(config, network)
        # train_episode resets before collecting; do not record the unused
        # construction-time episode as part of this training run.
        collector.recorder = recorder
        total_steps = 0
        for _ in range(config.episodes):
            result = train_episode(collector, optimizer)
            total_steps += result.steps
    return {"config": asdict(config), "resolved_device": str(device),
            "output": str(Path(output).resolve()), "episodes": config.episodes,
            "total_steps": total_steps, "last_training": asdict(result)}
