"""Explicit, read-only Thought probes for caller-selected representative samples."""

from dataclasses import dataclass, fields
import math

import torch
from torch.nn import functional as F

from self_genesis.batched_policy import BatchedObservation, BatchedPolicyState
from self_genesis.entity_memory import BatchedEntityMemory


@dataclass(frozen=True)
class ThoughtStepDiagnostics:
    step: int
    thought_norm: float
    change_norm: float
    cosine_similarity: float
    relative_change: float
    converged: bool
    saturation_fraction: float
    action_logits: tuple[float, ...]
    action_logit_changes: tuple[float, ...]
    value: float
    value_change: float


def _probe(network, inputs, memory, affect, retrieved, tolerance):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("convergence_tolerance must be finite and nonnegative")
    context = torch.cat((inputs, memory, affect, retrieved), dim=-1)
    previous = torch.zeros_like(memory)

    def readout(thought):
        # Every hypothetical update starts from the same incoming memory.
        candidate = network.memory_update(torch.cat((thought, affect), dim=-1), memory)
        logits = network.action_head(candidate).float()
        with torch.autocast(device_type=memory.device.type, enabled=False):
            value = network.value_head(candidate.to(network.value_head.weight.dtype)).squeeze(-1)
        return logits, value

    prior_logits, prior_value = readout(previous)
    result = []
    for step, thought in enumerate(network._think(context, memory), 1):
        current, prior = thought.float(), previous.float()
        norm = current.norm().item()
        change = (current - prior).norm().item()
        relative = change / max(prior.norm().item(), 1e-8)
        # ReLU saturation means inactive units; tanh saturation means |x| >= .99.
        saturated = current == 0 if network.thought_mode == "recurrent" else current.abs() >= .99
        logits, value = readout(thought)
        result.append(ThoughtStepDiagnostics(
            step, norm, change, F.cosine_similarity(current, prior, dim=-1, eps=1e-8).item(),
            relative, relative <= tolerance, saturated.float().mean().item(),
            tuple(logits.tolist()), tuple((logits - prior_logits).tolist()),
            value.item(), (value - prior_value).item()))
        previous, prior_logits, prior_value = thought, logits, value
    return tuple(result)


@torch.no_grad()
def probe_thought_steps(network, observation, state, *, communicating=False,
                        convergence_tolerance=1e-3):
    """Replay one scalar callback without sampling or writing any persistent state.

    Returns plain Python records, outside policy inputs and losses. Changes use
    the preceding step, with zero Thought as the step-one baseline (including a
    hypothetical memory readout). Cosine is zero for a zero vector. Convergence
    means change_norm / max(previous_norm, 1e-8) <= convergence_tolerance; it does
    not stop the loop. Action logits always describe NOTHING/GIVE, even when
    probing a communication callback. Shallow mode returns its single tanh step.
    """
    inputs = network._encode(observation, communicating)
    retrieved = network.retrieve_entity(observation.partner_appearance, state)
    return _probe(network, inputs, state.memory, state.affect, retrieved,
                  convergence_tolerance)


@torch.no_grad()
def probe_thought_steps_batch(network, observation, state, *, active, sample,
                              communicating=False, convergence_tolerance=1e-3):
    """Probe only sample=(world, observer), which must be active.

    Select a representative row before encoding/retrieval/recurrence, so trace
    cost is independent of batch size. The caller controls when and whom to probe.
    """
    if (len(sample) != 2 or any(type(i) is not int or i < 0 or i >= size
                               for i, size in zip(sample, state.memory.shape[:2]))):
        raise ValueError("sample must identify a valid (world, observer)")
    if (active.shape != state.memory.shape[:2] or active.dtype != torch.bool
            or active.device != state.memory.device):
        raise ValueError("active must be boolean [world, observer] on the policy device")
    world, observer = sample
    if not active[world, observer].item():
        raise ValueError("sample must be active")
    selection = (slice(world, world + 1), slice(observer, observer + 1))
    selected_observation = BatchedObservation(**{
        field.name: getattr(observation, field.name)[selection]
        for field in fields(BatchedObservation)})
    entities = BatchedEntityMemory(**{
        field.name: getattr(state.entities, field.name)[selection]
        for field in fields(BatchedEntityMemory)})
    selected_state = BatchedPolicyState(state.memory[selection], state.affect[selection], entities)
    selected_active = active[selection]
    inputs = network._encode_batch(selected_observation, selected_state, selected_active,
                                   communicating)[0]
    retrieved = entities.retrieve(selected_observation.partner_appearance)[0, 0]
    return _probe(network, inputs, selected_state.memory[0, 0], selected_state.affect[0, 0],
                  retrieved, convergence_tolerance)
