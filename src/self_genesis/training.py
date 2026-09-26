"""Complete-episode REINFORCE using only each agent's survival rewards."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.experiment import resolve_device
from self_genesis.observation import RunRecorder
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import Rollout, RolloutCollector


def survival_policy_loss(rollout: Rollout) -> torch.Tensor:
    """Sum decision losses, averaged over agents, with undiscounted returns.

    Each message (joint token log probability) and action receives only its
    owner's reward-to-go. Unselected and lone-survivor steps still contribute
    rewards. Recurrent graphs remain intact; discrete choices use score-function
    gradients. No baseline, social bonus, or auxiliary state target is used.
    Require complete episodes rather than silently zero-bootstrap time limits.
    """
    if (not rollout.terminated or rollout.truncated
            or any(items and items[0].step != 0 for items in rollout.experiences)):
        raise ValueError("Survival loss requires a complete episode from step zero")
    terms = []
    for experiences in rollout.experiences:
        reward_to_go = 0.0
        for experience in reversed(experiences):
            reward_to_go += experience.reward
            for decision in experience.decisions:
                if decision.log_prob is not None:
                    terms.append(-decision.log_prob * reward_to_go)
    if not terms:
        raise ValueError("Episode has no sampled policy decisions")
    return torch.stack(terms).sum() / len(rollout.experiences)


@dataclass(frozen=True)
class TrainingResult:
    loss: float
    steps: int
    survival_returns: tuple[float, ...]


def train_episode(collector: RolloutCollector,
                  optimizer: torch.optim.Optimizer) -> TrainingResult:
    """Reset, collect to extinction, and update the collector's shared network.

    Pass an optimizer for collector.network. Reset preserves sampling streams
    across episodes. Results contain no retained autograd graph.
    """
    collector.reset()
    # No agent can outlive its initial Life plus every Point in the world.
    config = collector.config
    rollout = collector.collect(config.initial_life + config.num_agents * config.initial_points)
    loss = survival_policy_loss(rollout)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    collector.detach()
    optimizer.step()
    result = TrainingResult(
        loss.item(), rollout.steps,
        tuple(sum(item.reward for item in items) for items in rollout.experiences))

    if collector.recorder is not None:
        collector.recorder.record_training(result, optimizer)
    return result


def run_training(config: ExperimentConfig, output: Path) -> dict:
    """Run a fixed number of complete updates and persist observations to JSONL."""
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    network = RecurrentPolicy(
        config.appearance_dim, vocabulary_size=config.vocabulary_size,
        max_message_length=config.max_message_length,
        memory_dim=config.memory_dim, affect_dim=config.affect_dim).to(device)
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
