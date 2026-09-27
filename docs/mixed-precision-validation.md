# Mixed-precision gradient validation

Measured on 2026-09-28 against base `c76eb0f493c60b32fe001f44e5b3504538281c6b`
(`dev/v0.0.9`), using Python 3.12, PyTorch 2.7.1+cu128, an NVIDIA RTX 5090,
and driver 595.91.07.

The original CUDA mixed-precision test failed for both Actor-Critic and REINFORCE
with messages of length 3, Entity Memory dimension 16, and a 32-step horizon.
The first failing `thought.weight` gradient errors were respectively 62.3745 and
42.7477, against bounds 41.8561 and 30.5315. The short, message-free cases passed.

Only the shared recurrent Linear/ReLU Thought core now disables autocast and
uses its parameter dtype (FP32 in supported BF16 training). Rounding in a shared
ReLU recurrence can change activation gates and their derivatives repeatedly;
the measured improvement below supports retaining this precision boundary.
Other eligible operations, including action/message heads, retain BF16 autocast.
Shallow Thought remains unchanged. FP64 reference execution retains FP64.
Autocast is restored before each diagnostic yield, so probe readouts retain the
same precision behavior as ordinary callbacks.

There is no detach, shortened recurrence, episode truncation, loss change, or
tolerance increase. Backward still traverses every configured Thought step and
the complete episode before an optimizer update.

## Matched measurements

The existing comparison uses initialization seed 18, world seeds 11 and 22,
fixed interior draws of 0.01, and two Adam updates at learning rate 1e-4.
Discrete choices and survival returns match FP32; losses retain their existing
`0.02 * abs(reference) + 0.002` bound. Parameter gradient error is the L2 norm
of the difference, with the unchanged bound `0.03 * reference.norm() + 0.002`.
These are gradients retained after backward, without gradient clipping.

For diagnosis, the gradient assertion was instrumented outside the committed
test to record all parameter errors instead of stopping at the first failure.
All other assertions remained active. Both versions measured 128 parameter
comparisons across the four cases and two updates. The normal tests separately
enforce every bound.

| Method | Horizon | Original maximum error / bound | Fixed maximum error / bound | Original → fixed violations |
| --- | ---: | ---: | ---: | ---: |
| Actor-Critic | 4 | 0.408797 | 0.180100 | 0 → 0 |
| Actor-Critic | 32 | 2.106975 | 0.482696 | 4 → 0 |
| REINFORCE | 4 | 0.494126 | 0.255037 | 0 → 0 |
| REINFORCE | 32 | 1.688502 | 0.461846 | 4 → 0 |

For the same `thought.bias` parameter on the first 32-step update, Actor-Critic
error fell from 15.521330 to 1.549331 (bound 7.366643), and REINFORCE error fell
from 9.062701 to 0.939959 (bound 5.367301). This isolates an improvement in the
same parameter, as the worst parameter can differ after the change.

## Validation commands

Run from the repository root with a CUDA-capable PyTorch environment:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_mixed_precision.py -v
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v
```

The added BF16 regression checks nonzero, finite gradients at every Thought step
at depths 16 and 32 and through the previous callback's Working Memory and
Affect. It also verifies FP32 Thought outputs and BF16 head outputs, including
readouts between generator yields. Existing full-episode execution tests cover
16/32/64 steps, scalar/batched execution, and CPU/CUDA.

These bounded measurements do not establish arbitrary-capacity or arbitrarily
long-episode BF16 stability. The FP32 Thought loop may reduce BF16 throughput;
no throughput improvement is claimed, and FP32 remains the default.

Final results: affected mixed-precision suite **6 passed** (6.885 s); full suite
with CUDA visible and the required cuBLAS setting **213 passed, no skips**
(100.797 s); full CPU suite **213 tests, OK with 15 CUDA-only skips** (78.921 s).
An initial full CUDA invocation omitted the required cuBLAS setting and produced
four deterministic-training setup errors; the corrected full rerun above passed.
`git diff --check` also passed.
