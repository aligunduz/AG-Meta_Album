# FO-Proto-LRSGMAML

Independent `baselines/fo_proto_lrsgmaml` submission: First-Order Proto-MAML
plus global Low-Rank Scalar-Gated Gradient Transport. Existing baseline files
are untouched. No task embedding, task-conditioned parameters, anchor,
router, or hypernetwork is used.

## Mathematical change

After the reference support-gradient clipping, for each eligible weight:

```text
G_matrix = stop_gradient(G).reshape(out_dim, -1)
G_tilde = reshape(sigmoid(a) * G_matrix + beta * U @ (V.T @ G_matrix), G.shape)
theta_fast = theta_fast - inner_lr * G_tilde
```

`low_rank_transport.py:transport_gradient` implements the only new mathematical
operation. `helpers_fo_proto_lrsgmaml.py:adapt` calls it immediately before the
reference `w - lr * g` update. Every persistent encoder tensor gets one global
scalar logit; weights named `weight` with at least two dimensions additionally
U and V of shape [weight.shape[0], min(rank, weight.shape[0])]. This covers ResNet Conv weights
and Linear weights. Bias and batch-normalization parameters get scalar gates
only. The task-local prototype W and b retain their ordinary FO-Proto-MAML
gradient updates without transport. They do not get U/V: their output axis represents arbitrary episode class indices and
changes with way (2–20 at test), rather than stable encoder output features.

The config defaults to rank 4, capped by the output dimension when necessary, fixed beta 1,
and logit 4, matching SGMAML's safe initialization: sigmoid(4) = 0.982014.
U starts as N(0, 0.01²), V as exact zeros.U starts as exact zeros and V is initialized from N(0, 1/C_out),
matching the original LRSGMAML initialization.
The residual therefore starts exactly zero.
At the first outer backward V has a connected but zero gradient because U=0;
U learns immediately, and V can learn on subsequent outer steps.

Support gradients use `create_graph=False`, `retain_graph=True`, followed by
explicit detach in transport. Thus there are no support Hessians, while query
loss differentiates through transport into a/U/V. The reference prototype
initialization graph and sibling fast-weight clones are preserved. LRSG lives
outside the encoder and fast weights, and is trained only by the same outer
Adam optimizer, including the reference per-task clipping and summed batches.

## Config, checkpoint and metrics

`config.json` is the native feedback configuration copied from FO-Proto-MAML.
Only the method identifier, transport flags and new `lrsg` section differ:

```json
"lrsg": {"enabled": true, "rank": 4, "beta": 1.0, "gate_init_logit": 4.0}
```

For exact disabled-transport comparison, set `lrsg.enabled` and both
`method_config.gradient_transport` / `method_config.low_rank_transport` false.
All other settings remain unchanged: ResNet18, seed 98, five inner steps,
encoder/classifier LR .01, outer Adam LR .001, clip 10, meta-batch 2,
30,000 training episodes, 300 validation episodes every 5,000 iterations,
and all train/validation sampling settings. Runner data seed remains 93.
Transforms, support/query BN, test episodes and output scoring are unchanged.

Best validation still uses pooled query accuracy and strict improvement.
`max-va.pth` stores the encoder and LRSG snapshots together with their ordered
architecture, config and score under method `fo-proto-lrsgmaml`. Loading checks
the architecture and strictly restores both states. Validation and testing
use learned transport during adaptation under `torch.enable_grad()`. As in the
reference, this is an inference checkpoint, without optimizer resumption.

Episode logging is unchanged. Since this runner has no epochs or native W&B
integration, aggregates are emitted every validation interval and for the last
partial interval to stdout and `<logger.logs_dir>/lrsg_metrics.jsonl`. If W&B is
already imported with an active run, they are also passed to `wandb.log`.
No run is created and W&B is not a required dependency.

Metrics: `lrsg/scalar_gate_mean`, `scalar_gate_min`, `scalar_gate_max`,
`low_rank_correction_norm`, `original_gradient_norm`,
`correction_to_gradient_ratio`, `u_norm`, `v_norm` (all with `lrsg/` prefix).
Gradient metrics are arithmetic means across eligible weight tensors and
training inner steps since the last log; validation is excluded. The ratio is
the mean of `||beta*LR(G)|| / (||G||+1e-12)` per observation, not a ratio of
means. G is the clipped support gradient actually transported. Gate statistics
and mean U/V Frobenius norms describe the current outer parameters.

## Run from repository root

```bash
python -u -m cdmetadl.run \
  --seed=93 \
  --input_data_dir=/content/meta_album_feedback \
  --submission_dir=/content/AG-Meta_Album/baselines/fo_proto_lrsgmaml \
  --output_dir_ingestion=/content/ag_meta_outputs/fo_proto_lrsgmaml/ingestion \
  --output_dir_scoring=/content/ag_meta_outputs/fo_proto_lrsgmaml/scoring \
  --test_tasks_per_dataset=100 \
  --overwrite_previous_results=False \
  --save_train_raw_outputs=False

python baselines/fo_proto_lrsgmaml/test.py \
  --checkpoint=/content/ag_meta_outputs/fo_proto_lrsgmaml/ingestion/model/max-va.pth \
  --input_data_dir=/content/meta_album_feedback --seed=93 --test_tasks_per_dataset=100

python -m unittest discover -s baselines/fo_proto_lrsgmaml/tests -v
```

`api.py`, `network.py`, `weight_names.py`, `metadata`, and `test.py` are exact
FO-Proto-MAML copies, allowing standalone submission imports. New logic lives
in `low_rank_transport.py`, `metrics.py`, the copied adaptation helper, and
the copied `model.py`. Tests cover formula/reshape, rank, initialization/RNG,
first-order behavior, query gradients including delayed U learning, optimizer
membership, fast-weight exclusion, disabled equivalence, variable-way eval,
checkpoint round-trip, logging, config parity and a real ResNet CPU backward.
Full dataset training and GPU benchmark results are not claimed.
