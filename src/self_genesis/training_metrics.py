"""Detached update diagnostics and explicitly scoped device measurements."""

import time

import torch

from self_genesis.batched_rollout import BatchedRollout
from self_genesis.world import Action


@torch.no_grad()
def rollout_metrics(rollout, network):
    """Decision means (including REINFORCE's untrained value predictions)."""
    device = next(network.parameters()).device
    totals = torch.zeros(7, device=device)

    def add(decision, target, *, active=None, communicating=False):
        if not communicating:
            if active is None:
                totals[0] += 1
                totals[1] += decision.choice == Action.GIVE
            else:
                totals[0] += active.sum()
                totals[1] += ((decision.choice == 1) & active).sum()
        if decision.log_prob is None:
            return
        value = decision.value.detach().float()
        entropy = decision.entropy.detach().float()
        if active is not None:
            value, entropy, target = value[active], entropy[active], target[active]
        totals[2] += (target - value).square().sum()
        totals[3] += value.numel()
        totals[4 if communicating else 5] += entropy.sum()
        if communicating:
            totals[6] += value.numel()

    if isinstance(rollout, BatchedRollout):
        returns = torch.zeros_like(rollout.experiences[0].reward, dtype=torch.float32)
        for experience in reversed(rollout.experiences):
            returns += experience.reward.float()
            for decision in experience.decisions:
                add(decision, returns, active=decision.active,
                    communicating=decision.communicating)
        survival = returns.mean().item()
    else:
        survival = 0.
        for experiences in rollout.experiences:
            target = 0.
            for experience in reversed(experiences):
                target += experience.reward
                for decision in experience.decisions:
                    add(decision, target, communicating=not isinstance(decision.choice, Action))
            survival += target / len(rollout.experiences)
    actions, gives, error, decisions, message_entropy, action_entropy, messages = totals.tolist()
    norms = [p.grad.detach().float().norm() for p in network.parameters() if p.grad is not None]
    return dict(action_callbacks=int(actions), encounters=int(actions) // 2,
                mean_observed_survival=survival, give_rate=gives / actions if actions else None,
                value_error=error / decisions if decisions else None,
                mean_action_entropy=action_entropy / actions if actions else None,
                mean_message_entropy=message_entropy / messages if messages else None,
                gradient_norm=torch.stack(norms).norm().item() if norms else 0.)


class UpdateMeasurement:
    """Wall time includes reset, collection, sampled logging and optimizer work."""

    def __init__(self, device):
        self.device = device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        self.start = time.perf_counter()

    def finish(self, world_steps):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self.start
        return dict(elapsed_seconds=elapsed, world_steps_per_second=world_steps / elapsed,
                    cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated(self.device)
                                               if self.device.type == "cuda" else None),
                    cuda_peak_reserved_bytes=(torch.cuda.max_memory_reserved(self.device)
                                              if self.device.type == "cuda" else None))


def measurement_metadata(device):
    return dict(device=str(device), gpu_name=(torch.cuda.get_device_name(device)
                if device.type == "cuda" else None), torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda, timing="perf_counter wall seconds",
                cuda_synchronized=device.type == "cuda",
                scope="reset through result construction, including sampled trace I/O, diagnostics "
                      "and scalar training record I/O; excludes measurement serialization",
                throughput_unit="completed world steps per second",
                memory_scope="process allocator peak during update, includes existing allocations",
                gpu_utilization=None, gpu_utilization_method="not sampled")
