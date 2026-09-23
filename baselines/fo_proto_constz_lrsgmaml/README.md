# FO-Proto-ConstZ-LRSGMAML

Independent control for `fo_proto_tclrsgmaml` in its LR-only setting. The
GateNet architecture and low-rank transport parametrization are retained,
but conditioning uses one learned shared vector instead of a support-derived
task embedding:

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

There is no support mean, task-embedding computation, or support/query input
to GateNet. The original support encoder forward is still required to
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
`list(transport.parameters())`, which includes z, scalar logits, U,V and
GateNet. The same Adam optimizer and outer gradient buffer handle all of them.
Neither adaptation nor validation/test modifies z.

## Configuration and initialization

```json
"constant_condition": {
  "enabled": true,
  "init": "zero"
}
```

Only this enabled zero initialization is supported. Mean-support or disabled
initialization raises an error. The LR-only GateNet settings, rank and beta
are validated. The legacy `task_conditioning` section and
`method_config.task_conditioned_gate=true` are retained to describe the active
GateNet/layout in the reference configuration; they do not enable data-derived
conditioning in this baseline. The reference's checked-in scalar scale is 1;
this control explicitly sets it to the requested LR-only value 0.

At z=0, the unchanged zero-initialized output layer produces delta_c=0.
Initially U=0 also blocks the low-rank conditioning gradient: z has a connected
but zero query-loss gradient, and the initial update matches scalar-gated
LRSG. U learns first, then the low-rank GateNet output becomes active, allowing
subsequent query losses to train z and the hidden layer. Tests check both the
initial zero gradient and nonzero z gradients/updates after activation using
synthetic episodes and the default initialization.

## Checkpoints and logging

The independent method identifier is `fo-proto-constz-lrsgmaml`.
`max-va.pth` retains best-validation selection and stores z under
`state.lrsg["z"]` alongside logits/U/V and `gate_net.*`. Architecture metadata
also records the constant condition and `[512]` shape. Strict loading rejects
missing z or incompatible architecture. Optimizer-state resumption is not
implemented, matching the reference.

Existing logging is preserved. The `tc/` metric names remain for compatibility.
Scalar residual metrics are zero. Low-rank task variation is zero for an
unchanged outer state; aggregation across optimizer updates can show variation
because the shared parameters have changed.

## Unit/synthetic validation

```text
python -B -m unittest discover -s baselines/fo_proto_constz_lrsgmaml/tests -v
```

13 tests pass on CPU, covering identical GateNet inputs across different
episodes, bitwise identical transport for the same G, query-to-z gradients,
outer Adam updates from default initialization, unchanged z during adaptation,
first-order behavior, untouched W,b updates, zero scalar residual/gradient,
GateNet/layout and reference-file equality, the Conv transport formula,
config parity, synthetic meta-batching, checkpoint round-trip/rejection,
2/7/20-way evaluation, logging and ResNet18 five-step synthetic backward.
No full Meta-Album training or real-data evaluation was run.

Files: `model.py`, `task_transport.py`,
`helpers_fo_proto_constz_lrsgmaml.py`, `config.json`,
`tests/test_fo_proto_constz_lrsgmaml.py`, this README, and unchanged local
copies of `api.py`, `network.py`, `weight_names.py`, `metadata`, `metrics.py`,
`test.py`, `gate_net.py`, `low_rank_transport.py`. Runtime code imports only
local copies; it does not depend on another baseline directory. The TC
evaluation ablation utility is not copied because its data-derived conditions
are outside the scope of this control.

This is a control for the value of task-specific conditioning while retaining
GateNet capacity and optimization. Its learned coefficients are globally
shared, so it cannot choose different coefficients from episode content.
Whether task-specific conditioning improves accuracy still requires the
separate empirical comparison.
