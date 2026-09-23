# FO-Proto-ConstZ-LRSGMAML

Independent control for `fo_proto_tclrsgmaml` in its LR-only setting. The
GateNet architecture and low-rank transport parametrization are retained,
but conditioning uses either one learned shared vector (`init="zero"`) or
an EMA of completed training episodes (`init="ema"`). The current task's
support embedding is never used for its own conditioning. In zero mode:

```text
z = nn.Parameter(zeros(512))
delta_c = GateNet(z)[rank_output_slices]
G_tilde = sigmoid(a) * G + reshape(beta * U @ diag(1 + delta_c) @ V.T @ G_matrix)
```

`ConstantConditionedTransport.z` in `task_transport.py` is an outer/meta
parameter. `condition()` accepts no episode argument and calls
`self.gate_net(self.z)` once per episode, without detaching z. Its residuals
are reused for all inner steps. At a fixed outer parameter state, every task
receives exactly the same GateNet input and coefficients. The transport
output is therefore identical for identical G; actual support gradients and
adapted classifiers may still differ between tasks. Outer updates can change z
between meta-batches.

In zero mode there is no support mean, task-embedding computation, or
support/query input to GateNet. The original support encoder forward is required to
initialize the Proto classifier W,b. Those prototypes retain their original
encoder gradient connection. W,b receive ordinary clipped first-order updates
and are never passed through transport.

## Preserved controls

- GateNet: Linear(512,128), ReLU, Linear(128,output_size); input norm `none`.
- The output layout remains `N + sum(layer_ranks)`: the first N outputs are
  retained but multiplied by `scalar_delta_scale=0`; low-rank outputs use
  `low_rank_delta_scale=1`. Scalar output rows therefore receive zero gradient.
- Rank 4, capped at each layer's output width; beta 1; scalar logits initially
  4; U initially zero; V uses the reference private RNG and initialization.
- Scalar a and U,V remain learned outer parameters. The low-rank formula,
  factor normalization behavior and initialization are byte-identical copies
  of the TC baseline's `low_rank_transport.py`.
- First order: support gradients use `create_graph=False, retain_graph=True`
  and transport detaches G. Only encoder and task-local W,b are fast weights.
- ResNet18, five inner steps, encoder/classifier LR .01, outer Adam LR .001,
  clipping 10, summed meta-batches of two, seeds and episode protocols remain
  unchanged. Both inner gradient clipping and per-task outer clipping retain
  the reference behavior.

`model.py` builds `meta_parameters` as encoder parameters plus
`list(transport.parameters())`, which includes scalar logits, U,V and GateNet,
plus z in zero mode. The same Adam optimizer and outer gradient buffer handle
all of them. Neither adaptation nor validation/test modifies z. EMA mode has
no z parameter; its buffers are excluded from the optimizer.

## Configuration and initialization

```json
"constant_condition": {
  "enabled": true,
  "init": "ema",
  "ema_decay": 0.99
}
```

The checked-in config selects EMA. Set `init="zero"` to retain the original
learned-z behavior; `ema_decay` is unused and optional in zero mode. EMA
requires a finite numeric decay in `[0, 1)`. Other initialization modes and
disabled conditioning raise an error. The LR-only GateNet settings, rank and
beta are validated. The legacy `task_conditioning` section and
`method_config.task_conditioned_gate=true` are retained to describe the active
GateNet/layout in the reference configuration; they do not enable current-task
conditioning. The reference's checked-in scalar scale is 1;
this control explicitly sets it to the requested LR-only value 0.

At z=0, the unchanged zero-initialized output layer produces delta_c=0.
Initially U=0 also blocks the low-rank conditioning gradient: z has a connected
but zero query-loss gradient, and the initial update matches scalar-gated
LRSG. U learns first, then the low-rank GateNet output becomes active, allowing
subsequent query losses to train z and the hidden layer. Tests check both the
initial zero gradient and nonzero z gradients/updates after activation using
synthetic episodes and the default initialization.

## EMA episode order and isolation

`task_transport.py` registers `m` (shape `[512]`) and `m_initialized` as
persistent buffers. They have no gradient and are never fast weights or
optimizer parameters. For each training episode t:

1. The initial prototype support forward also provides
   `e_t = features.detach().mean(dim=0)`. No extra encoder forward is added.
2. `condition()` uses only a detached clone of the previously committed m:
   `delta_c = GateNet(m_t)[rank_output_slices]`. Cloning prevents later buffer
   updates from mutating the tensor saved for this episode's backward pass.
3. Adaptation, query loss, backward and outer gradient bookkeeping finish.
4. `meta_fit` calls `update_ema(e_t)` once for this completed training episode:
   `m <- decay * m + (1 - decay) * e_t`. This happens per episode, including
   episodes within the same meta-batch, before validation/checkpoint selection.

On the first training episode, GateNet is bypassed and all residuals are
exactly zero, even if GateNet already has nonzero biases. This is LRSG
equivalence, including when U is nonzero. Zero-valued connections to GateNet
parameters preserve the outer loop's gradient-presence contract without
training GateNet on this episode. At the end of the episode, m is initialized
with `e_t` directly and `m_initialized` becomes true. EMA is first supplied to
GateNet on the next episode.

The helper returns the detached support mean to the training owner only when
requested; it never updates the buffer. No pending task embedding is stored
on the transport module. Validation/test neither computes an EMA support mean
nor calls the update; `update_ema` also guards against updates in eval mode.
They use the latest saved training EMA unchanged. Evaluation of an
uninitialized EMA uses zero residuals and does not initialize from test data.

## Checkpoints and logging

The independent method identifier is `fo-proto-constz-lrsgmaml`.
`max-va.pth` retains best-validation selection and stores z under
`state.lrsg["z"]` in zero mode, or `state.lrsg["m"]` and
`state.lrsg["m_initialized"]` in EMA mode, alongside logits/U/V and `gate_net.*`.
Architecture metadata records the mode, `[512]` shape and EMA decay when used.
The original zero-mode state keys and architecture remain compatible. Strict
loading rejects missing state or incompatible architecture. Test uses the
training EMA saved with the selected best-validation checkpoint; later
training state is not mixed into that snapshot. Optimizer-state resumption
is not implemented, matching the reference.

Existing logging is preserved. The `tc/` metric names remain for compatibility.
Scalar residual metrics are zero. Low-rank task variation is zero for an
unchanged outer state in zero mode; aggregation across optimizer updates can
show variation because the shared parameters have changed. In EMA mode,
completed training episodes can also change the conditioning input.

## Unit/synthetic validation

```text
python -B -m unittest discover -s baselines/fo_proto_constz_lrsgmaml/tests -v
```

20 tests pass on CPU. The original 13 zero-mode tests cover identical GateNet inputs across different
episodes, bitwise identical transport for the same G, query-to-z gradients,
outer Adam updates from default initialization, unchanged z during adaptation,
first-order behavior, untouched W,b updates, zero scalar residual/gradient,
GateNet/layout and reference-file equality, the Conv transport formula,
config parity, synthetic meta-batching, checkpoint round-trip/rejection,
2/7/20-way evaluation, logging and ResNet18 five-step synthetic backward.
Seven EMA tests additionally cover no current-task leakage, exact episode
ordering and recurrence, first-episode LRSG equivalence, detached buffers,
optimizer exclusion, immutable backward inputs, eval isolation, checkpoint
round-trip, config validation and TC-LR architecture/formula equivalence.
No full Meta-Album training or real-data evaluation was run.

Files: `model.py`, `task_transport.py`,
`helpers_fo_proto_constz_lrsgmaml.py`, `config.json`,
`tests/test_fo_proto_constz_lrsgmaml.py`, `tests/test_ema_condition.py`, this README, and unchanged local
copies of `api.py`, `network.py`, `weight_names.py`, `metadata`, `metrics.py`,
`test.py`, `gate_net.py`, `low_rank_transport.py`. Runtime code imports only
local copies; it does not depend on another baseline directory. The TC
evaluation ablation utility is not copied because its data-derived conditions
are outside the scope of this control.

This is a control for the value of task-specific conditioning while retaining
GateNet capacity. Zero mode learns a global conditioning parameter; EMA mode
uses training history and is therefore sensitive to training episode order.
Neither chooses coefficients from the current episode's content. Whether
task-specific conditioning improves accuracy requires a separate empirical
comparison.
