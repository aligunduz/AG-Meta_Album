# FO-Proto-TCLRSGMAML

Independent submission extending the current FO-Proto-LRSGMAML with support-only
task conditioning. All code is local to this directory; existing FO-Proto-MAML,
FO-Proto-TCSGMAML and FO-Proto-LRSGMAML files are unchanged.

## Algorithm

```text
G_matrix = stop_gradient(clipped_support_gradient).reshape(C_out, -1)
G_tilde_tau = sigmoid(a + delta_a_tau) * G
              + reshape(beta * U @ diag(1 + delta_c_tau) @ V.T @ G_matrix)
theta_fast = theta_fast - inner_lr * G_tilde_tau
```

The implementation scales rows of `V.T @ G_matrix` instead of materializing
the diagonal. `a, U, V` are global/shared outer parameters. Only `delta_a_tau`
and `delta_c_tau` depend on the support set. Encoder Conv/Linear weights get
both terms; encoder bias/BN/1D parameters get the scalar term only. Prototype
classifier W and b receive ordinary clipped FO-Proto-MAML gradient updates,
without any transport. Strict zip checks prevent accidental head truncation.

Task embedding is `support_features.mean(dim=0).detach()`, from the exact same
initial encoder forward that initializes the prototypes. GateNet follows
FO-Proto-TCSGMAML: Linear → ReLU → Linear, with an optional input normalization
and default `none`. Its final weight and bias are zero. Hidden initialization
uses the same RNG-isolated construction as TCSG. The network runs once per
episode; residuals are reused across all inner steps and never stored as
persistent task state. Query features/labels are not conditioning inputs.

## Output layout and initialization

Let N be the number of encoder parameter tensors. Outputs `[0:N]` give one
scalar residual per tensor in `encoder.named_parameters()` registration order,
scaled by `scalar_delta_scale`. Remaining outputs give consecutive rank
residual vectors in that same order for eligible weights only, scaled by
`low_rank_delta_scale`. Each rank is `min(config_rank, C_out)`. The explicit
`rank_layout` records name, parameter key, rank and absolute start/stop offsets.
The network output size is `N + sum(layer_ranks)`.

U/V initialization is copied exactly from current LRSG: U=0 and
V ~ N(0, 1/C_out), where 1/C_out is the variance. A private CPU generator seeded
98 creates V, preserving the surrounding RNG state. Global gate logits start
at 4 (sigmoid ≈ .982014), rank defaults to 4 and beta is fixed at 1. No L2,
column, spectral normalization, orthogonalization or unit-norm constraint is
applied to either U or V.

The coefficient is **1 + delta_c**, not delta_c. Therefore zero residuals give
the original LRSG operator even with learned, nonzero U/V. CPU float64 tests
confirm bitwise-identical fast weights, query logits and encoder outer
gradients against the current LRSG helper, both at initialization and with
nonzero U. No behavior outside task conditioning is changed: backbone, rank,
U/V initialization, optimizer, learning rates, clipping, seeds, data and
training/validation/test protocols and best-checkpoint selection are retained.

Support autograd uses `create_graph=False, retain_graph=True`; transport
explicitly detaches G. Thus support Hessians are absent while query loss
trains the encoder, global logits, U, V and GateNet through the fast updates.
Prototype initialization remains connected to the encoder. LRSG and GateNet
parameters never enter fast weights and only the outer Adam updates them.
At U=0 the initial V/rank-residual gradients are connected but zero. Likewise,
the zero final GateNet layer initially gives zero hidden-layer gradients.
U and scalar outputs learn first; subsequent steps activate the other paths.

## Configuration and checkpoint

`config.json` copies LRSG's complete feedback config. Only the method name,
task-conditioned flag and this section differ:

```json
"task_conditioning": {
  "enabled": true,
  "hidden_size": 128,
  "input_norm": "none",
  "scalar_delta_scale": 1.0,
  "low_rank_delta_scale": 1.0
}
```

To disable conditioning, set both `task_conditioning.enabled` and
`method_config.task_conditioned_gate` false. This removes GateNet parameters
and recovers global LRSG. To disable all transport additionally set
`lrsg.enabled`, `method_config.gradient_transport` and
`method_config.low_rank_transport` false.

The runner retains model seed 98, data seed 93, ResNet18, five inner steps,
encoder/classifier LR .01, Adam LR .001, clip 10, summed meta-batches of two,
30,000 training episodes and 300 validation episodes every 5,000 iterations.

`max-va.pth` uses method `fo-proto-tclrsgmaml`, format_version 1, full config,
model arguments and best validation score. `state.encoder` stores the encoder;
`state.lrsg` stores global logits/U/V plus `gate_net.*`; `state.architecture`
stores encoder names/shapes, rank, beta, enable flags, scales, GateNet structure
and the exact scalar/rank output mapping. Loading rejects architecture
mismatches and loads all state strictly. Validation/test use the learned
GateNet to compute fresh coefficients from each episode's support set.
As in LRSG, checkpoints do not implement optimizer-state training resumption.

## Aggregate logging

The original episode logger is unchanged. Every validation interval, plus the
final partial interval, `metrics.py` logs to stdout and
`<logger.logs_dir>/lrsg_metrics.jsonl`, and to W&B if a run is already active.
It does not create a W&B run or introduce a W&B dependency.

The eight `lrsg/` metrics are retained: `scalar_gate_mean/min/max`,
`low_rank_correction_norm`, `original_gradient_norm`,
`correction_to_gradient_ratio`, `u_norm`, `v_norm`. Gate statistics now describe
actual conditioned gates across training episodes. Gradient diagnostics use
the actual clipped gradients and conditioned correction, averaged over
eligible tensors and training inner steps; the ratio includes beta.
U/V norms remain means over layers at the current outer snapshot.

New `tc/` diagnostics include `scalar_delta_mean/std/abs_mean`,
`low_rank_delta_mean/std/abs_mean`, and `low_rank_coeff_mean/std`. These moments
aggregate residual components across training episodes, once per episode.
`scalar_delta_task_std_mean` and `low_rank_delta_task_std_mean` additionally
compute each coordinate's population std across episodes and average those
stds, separating task variation from differences between layers/ranks.
Validation/test do not pollute training aggregates; logs reset the accumulators.

Because U/V are not normalized, delta_c/c should primarily be interpreted
through variation across tasks, not as an absolute directional magnitude
independent of U and V. Inspect mean, std and task-to-task variation alongside
`lrsg/correction_to_gradient_ratio`. Variation over a training interval can
also reflect outer parameter updates, not only differences between tasks.

## Run and verify

```bash
python -u -m cdmetadl.run \
  --seed=93 \
  --input_data_dir=/content/meta_album_feedback \
  --submission_dir=/content/AG-Meta_Album/baselines/fo_proto_tclrsgmaml \
  --output_dir_ingestion=/content/ag_meta_outputs/fo_proto_tclrsgmaml/ingestion \
  --output_dir_scoring=/content/ag_meta_outputs/fo_proto_tclrsgmaml/scoring \
  --test_tasks_per_dataset=100 \
  --overwrite_previous_results=False \
  --save_train_raw_outputs=False

python baselines/fo_proto_tclrsgmaml/test.py \
  --checkpoint=/content/ag_meta_outputs/fo_proto_tclrsgmaml/ingestion/model/max-va.pth \
  --input_data_dir=/content/meta_album_feedback --seed=93 --test_tasks_per_dataset=100

python -m unittest discover -s baselines/fo_proto_tclrsgmaml/tests -v
```

Tests cover zero-residual equivalence, nonzero scalar/rank formulas, capped
rank layout, Conv reshape, ungated head updates, support-only detached
conditioning, episode isolation, FO and outer gradients, optimizer membership,
2/7/20-way evaluation, strict checkpoint round-trip/mismatch rejection, metrics,
config parity and real ResNet CPU backward after five inner steps.

Files: `model.py` (training/checkpoint), `helpers_fo_proto_tclrsgmaml.py`
(adaptation), `low_rank_transport.py` (LRSG formula), `task_transport.py`
(conditioning/mapping/metrics), `gate_net.py` (TCSG GateNet), `config.json`,
`metrics.py`, `test.py`, tests and this README. `api.py`, `network.py`,
`weight_names.py`, `metadata`, `test.py` and `metrics.py` are unchanged local
copies from LRSG for self-contained submission loading.

No full dataset training or GPU experiment was run. Existing FO-Proto-MAML
10/10 and FO-Proto-TCSGMAML 12/12 tests pass. The untouched LRSG suite passes
16/17; its metrics test fails on a 5.96e-8 float32 matrix-association rounding
difference at a seven-decimal assertion. Existing baseline files are unchanged.
