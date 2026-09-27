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
            "affect": state.affect.detach().cpu().tolist(),
            "entity_memory": _entities(state)}


def _entities(state):
    return [{"appearance": entry.appearance.detach().cpu().tolist(),
             "value": entry.value.detach().cpu().tolist()} for entry in state.entities]


def entity_memory_record(network, appearance, before, after):
    """Detached analysis snapshot; never passed back to a policy or memory writer."""
    return {
        "appearance": appearance.detach().cpu().tolist(),
        "retrieved": network.retrieve_entity(appearance, before).detach().cpu().tolist(),
        "state_before": _entities(before),
        "state_after": _entities(after),
    }


def _resources(world):
    return {"life": world.state.life.tolist(), "points": world.state.points.tolist()}


class RunRecorder:
    """One exclusive output file per collector; records never retain graphs.

    Agent numbers and episode numbers are logging keys only. A summary is
    cumulative within its episode; survivors are right-censored at that boundary.
    """

    def __init__(self, path: str | Path, *, training: bool = False):
        self._file = Path(path).open("x", encoding="utf-8")
        self.training = training
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

    def samples_step(self, collector):
        config = collector.config
        return (not self.training or (0 in config.trace_worlds
                and self.episode % config.trace_update_interval == 0
                and collector.elapsed_steps % config.trace_step_interval == 0))

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
            parameter_count=network.parameter_count,
            policy_settings={name: getattr(network, name) for name in (
                "appearance_dim", "vocabulary_size", "max_message_length",
                "memory_dim", "affect_dim", "entity_memory_dim")},
            **_resources(collector.world),
            appearance=collector.world.state.appearance.tolist(),
            point_generation_probability=(
                collector.world.state.point_generation_probability.tolist()),
            states=([_state(agent.state) for agent in collector.agents]
                    if self.samples_step(collector) else []),
            trace_complete=(not self.training or (self.samples_step(collector)
                            and collector.config.trace_step_interval == 1)))

    def record_step(self, collector, result, policies):
        for policy in policies:
            for decision in policy.decisions:
                if isinstance(decision.choice, Action):
                    self.actions[decision.choice.value] += 1
                else:
                    for token in decision.choice:
                        self.tokens[token] += 1
        for index, reward in enumerate(result.reward.tolist()):
            self.lifetimes[index] += reward
            if result.died[index]:
                self.death_steps[index] = collector.elapsed_steps
        if not self.samples_step(collector):
            return
        callbacks = []
        participants = sorted(
            (index for index, policy in enumerate(policies) if policy.decisions),
            key=lambda index: not policies[index].decisions[0].observation.first)
        # Group messages before actions, preserving each pair's dependency order.
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
                else:
                    choice = list(decision.choice)
                callbacks.append({
                    "agent": index, "phase": "message" if phase == 0 else "action",
                    "choice": choice, "observation": observation,
                    "log_probability": (decision.log_prob.detach().item()
                                        if decision.log_prob is not None else None),
                    "entity_memory": entity_memory_record(
                        collector.network, obs.partner_appearance,
                        decision.state_before, decision.state_after),
                    "state_before": _state(decision.state_before),
                    "state_after": _state(decision.state_after),
                })
        self._write(
            "step", step=collector.elapsed_steps, participants=participants,
            pairs=collector.protocol.last_pairs,
            callbacks=callbacks,
            entity_memory_completions=[dict(agent=index, **policies[index].completion)
                                       for index in participants
                                       if policies[index].completion is not None],
            rewards=result.reward.tolist(), died=result.died.tolist(),
            generated_points=result.generated_points.tolist(),
            successful_transfers=[{"donor": donor, "recipient": recipient}
                                  for donor, recipient in result.successful_transfers],
            **_resources(collector.world),
            states=[_state(agent.state) for agent in collector.agents])

    def record_summary(self, collector, *, max_steps, steps, terminated,
                       horizon_completed=False):
        completed = [age for age, death in zip(self.lifetimes, self.death_steps)
                     if death is not None]
        total_actions = sum(self.actions.values())
        self._write(
            "summary", elapsed_steps=collector.elapsed_steps,
            collection_max_steps=max_steps, collection_steps=steps,
            terminated=terminated, truncated=not (terminated or horizon_completed),
            horizon_completed=horizon_completed,
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
            "training", **asdict(result),
            optimizer=type(optimizer).__qualname__,
            optimizer_settings=[{key: value for key, value in group.items() if key != "params"}
                                for group in optimizer.param_groups])


class BatchedTraceRecorder:
    """Serialize only selected worlds/steps; never retain tensor graphs."""

    def __init__(self, config, write):
        self.config = config
        self.write = write
        self.update = 0

    def record_step(self, collector, result, previous_steps):
        config = self.config
        if not config.trace_worlds or self.update % config.trace_update_interval:
            return
        for world in config.trace_worlds:
            # Done worlds freeze; do not duplicate their final snapshot.
            step = int(previous_steps[world])
            if (int(collector.world.steps[world]) == step
                    or step % config.trace_step_interval):
                continue

            def row(tensor):
                return tensor[world].detach().cpu().tolist()

            state = result.state
            self.write(
                "batch_trace", update=self.update, world=world, step=step,
                participants=row(result.pairs),
                decisions=[dict(communicating=d.communicating, active=row(d.active),
                                choice=row(d.choice),
                                observation={name: row(value) for name, value
                                             in vars(d.observation).items()})
                           for d in result.decisions],
                rewards=row(result.world.reward), died=row(result.world.died),
                generated_points=row(result.world.generated_points),
                successful_transfers=row(result.world.successful_transfers),
                life=row(collector.world.state.life), points=row(collector.world.state.points),
                appearance=row(collector.world.state.appearance),
                working_memory=row(state.memory), affect=row(state.affect),
                entity_memory={name: row(value) for name, value in vars(state.entities).items()})
