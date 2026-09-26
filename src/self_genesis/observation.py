"""Streaming JSONL experiment records, separate from policy observations."""

from dataclasses import asdict
import json
from pathlib import Path

import torch

from self_genesis.world import Action


def _json_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Unsupported record value: {type(value).__name__}")


def _state(state):
    return {"working_memory": state.memory.detach().cpu().tolist(),
            "affect": state.affect.detach().cpu().tolist()}


def _resources(world):
    return {"life": world.state.life.tolist(), "points": world.state.points.tolist()}


class RunRecorder:
    """One exclusive output file per collector; records never retain graphs.

    Agent numbers and episode numbers are logging keys only. A summary is
    cumulative within its episode; survivors are right-censored at that boundary.
    """

    def __init__(self, path: str | Path):
        self._file = Path(path).open("x", encoding="utf-8")
        self.episode = -1
        self._collector = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._file.close()

    def _write(self, kind, **values):
        self._file.write(json.dumps(
            {"schema_version": 1, "type": kind, "episode": self.episode, **values},
            allow_nan=False, default=_json_tensor) + "\n")
        self._file.flush()

    def start_episode(self, collector):
        if self._collector is not None and self._collector is not collector:
            raise ValueError("Use a separate RunRecorder for each collector")
        self._collector = collector
        self.episode += 1
        self.lifetimes = [0.0] * collector.config.num_agents
        self.death_steps = [None] * collector.config.num_agents
        self.actions = {action.value: 0 for action in Action}
        self.tokens = [0] * collector.network.vocabulary_size
        network = collector.network
        self._write(
            "episode_start", settings=asdict(collector.config),
            resolved_device=str(collector.world.state.life.device),
            policy_settings={name: getattr(network, name) for name in (
                "appearance_dim", "vocabulary_size", "max_message_length",
                "memory_dim", "affect_dim")},
            **_resources(collector.world),
            appearance=collector.world.state.appearance.tolist(),
            states=[_state(agent.state) for agent in collector.agents])

    def record_step(self, collector, result, policies):
        callbacks = []
        participants = sorted(
            (index for index, policy in enumerate(policies) if policy.decisions),
            key=lambda index: not policies[index].decisions[0].observation.first)
        # Restore protocol order: both messages, then both actions.
        for phase in range(2):
            for index in participants:
                decision = policies[index].decisions[phase]
                obs = decision.observation
                observation = {
                    "life": obs.life, "points": obs.points,
                    "partner_life": obs.partner_life, "partner_points": obs.partner_points,
                    "partner_appearance": obs.partner_appearance.tolist(),
                    "first": obs.first, "received_message": list(obs.received_message),
                    "partner_action": obs.partner_action.value if obs.partner_action else None,
                }
                if isinstance(decision.choice, Action):
                    choice = decision.choice.value
                    self.actions[choice] += 1
                else:
                    choice = list(decision.choice)
                    for token in choice:
                        self.tokens[token] += 1
                callbacks.append({
                    "agent": index, "phase": "message" if phase == 0 else "action",
                    "choice": choice, "observation": observation,
                    "log_probability": (decision.log_prob.detach().item()
                                        if decision.log_prob is not None else None),
                    "state_before": _state(decision.state_before),
                    "state_after": _state(decision.state_after),
                })
        for index, reward in enumerate(result.reward.tolist()):
            self.lifetimes[index] += reward
            if result.died[index]:
                self.death_steps[index] = collector.elapsed_steps
        self._write(
            "step", step=collector.elapsed_steps, participants=participants,
            callbacks=callbacks, rewards=result.reward.tolist(), died=result.died.tolist(),
            **_resources(collector.world),
            states=[_state(agent.state) for agent in collector.agents])

    def record_summary(self, collector, *, max_steps, steps, terminated):
        completed = [age for age, death in zip(self.lifetimes, self.death_steps)
                     if death is not None]
        total_actions = sum(self.actions.values())
        self._write(
            "summary", elapsed_steps=collector.elapsed_steps,
            collection_max_steps=max_steps, collection_steps=steps,
            terminated=terminated, truncated=not terminated,
            lifetimes=[{"agent": index, "observed_steps": age,
                        "death_step": self.death_steps[index],
                        "censored": self.death_steps[index] is None}
                       for index, age in enumerate(self.lifetimes)],
            deaths=len(completed),
            mean_survival_time=(sum(self.lifetimes) / len(self.lifetimes)
                                if terminated else None),
            mean_completed_lifetime=sum(completed) / len(completed) if completed else None,
            mean_observed_lifetime=sum(self.lifetimes) / len(self.lifetimes),
            action_counts=self.actions,
            action_ratios={key: value / total_actions if total_actions else None
                           for key, value in self.actions.items()},
            token_counts=self.tokens, **_resources(collector.world))

    def record_training(self, result, optimizer):
        self._write(
            "training", loss=result.loss, steps=result.steps,
            survival_returns=result.survival_returns,
            optimizer=type(optimizer).__qualname__,
            optimizer_settings=[{key: value for key, value in group.items() if key != "params"}
                                for group in optimizer.param_groups])
