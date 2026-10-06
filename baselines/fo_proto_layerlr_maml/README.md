# FO-Proto-MAML + learned per-tensor inner step size

**FO-Proto-MAML with one learned, unbounded inner step size per encoder tensor; no gradient transform.**

Submission key: `fo_proto_layerlr_maml`; checkpoint method: `fo-proto-layerlr-maml`.

## Why this baseline exists

The direction x magnitude decomposition of the LR-only TC checkpoints
(`fo_proto_tclrsgmaml/eval_direction_magnitude.py`, data seeds 93 / 97 / 101)
attributes the gain of the low-rank transform to the per-tensor SIZE of its
update, not to its direction. In that method the scalar gate is `sigmoid(a) <= 1`
and the base step is fixed, so the low-rank path is the only way the model can
enlarge a step. This baseline gives FO-Proto-MAML the direct alternative: a
learned step size per encoder tensor that may exceed the base step, with the raw
gradient direction. It answers whether the low-rank transform is needed once
per-tensor step sizes can be learned.

It is a separately trained control, not the `mag_layer` diagnostic: here the
multiplier is one constant per tensor (shared by all tasks), and the encoder is
trained together with it.

## Update rule

For encoder tensor `j` and inner step `s`:

```text
alpha[s, j] = encoder_lr * exp(log_scale[s, j])
theta_next[j] = theta[j] - alpha[s, j] * g[j]
```

- `g` is the clipped, detached support gradient (unchanged FO-Proto-MAML).
- `log_scale` is initialized to zeros: no RNG is consumed and the first
  adaptation is exactly FO-Proto-MAML. There is no upper or lower bound.
- `inner_lr.per_step=false` (default): one multiplier per tensor, shared by the
  five inner steps, i.e. 60 learned scalars for ResNet18 (20 convolution weights
  and 40 BN affine vectors). `per_step=true` stores one row per inner step.
- The two task-local prototype classifier tensors keep the fixed step 0.01.
- No low-rank residual, scalar gate, GateNet, EMA, task embedding or curvature.

The step sizes are not detached. With first-order support gradients the query
loss trains them through `theta_K[j] = theta_0[j] - alpha[j] * sum_s g_s[j]`:

```text
dL_query / dlog_scale[j] = -alpha[j] * < dL_query/dtheta_K[j], sum_s g_s[j] >
```

`log_scale` joins the encoder in the same Adam (0.001), the same per-task
elementwise clipping (10) and the same summed meta-batch (2).

The exponential form makes growth multiplicative: Adam moves `log_scale` by at
most about 0.001 per update, so the multiplier can reach `exp(0.001 * updates)`,
about 12x after 5,000 training tasks and 148x after 10,000. Check the logged
multipliers before concluding: if they are still rising at that limit at the
selected checkpoint, the comparison was limited by this rate, not by the method.

## Preserved settings

`config.json` is `fo_proto_maml/config.json` plus the method identifier,
`method_config.learned_inner_lr=true` and the `inner_lr` section. `api.py`,
`network.py`, `weight_names.py` and `test.py` are byte-identical copies from
`fo_proto_maml`. No existing baseline or ingestion/scoring code is modified.

| Setting | Value |
| --- | --- |
| Encoder | ResNet18, pretrained=False; same initialization and RNG order |
| Model seed | 98 |
| Inner steps; base encoder / classifier step | 5; 0.01 / 0.01 |
| Outer optimizer / LR | Adam / 0.001, encoder and `log_scale` together |
| Meta-batch | 2 tasks, sum gradients, no averaging |
| Clipping | elementwise +-10 on support gradients; per-task outer gradients before the sum |
| Training budget | 30,000 tasks; validation on 300 tasks every 5,000 |
| Train / validation episodes | 5-way 10-shot / 5-way 5-shot, 20 queries per class |
| Test episodes | unchanged runner: 2-20 ways, 1-20 shots, 20 queries per class |
| Checkpoint selection | strict improvement of pooled validation accuracy, `max-va.pth` |

Train, validation and test use the same update rule; there is no evaluation
multiplier (no gamma policy).

## Logs and checkpoint

At every validation interval `inner_lr_metrics.jsonl` (in the runner's logs
directory), stdout and an already-active W&B run receive the multipliers
`exp(log_scale)`: `inner_lr/scale_min`, `scale_geomean`, `scale_max`,
`scale_geomean_weight` (tensors with ndim >= 2) and `scale_geomean_vector`
(BN affine vectors). Geometric means are unweighted by tensor size. The JSONL
file additionally stores every per-tensor multiplier under `scales`.

`max-va.pth` stores the selected encoder, `log_scale` and a layout (ordered
tensor names and shapes, step count, `per_step`, base step, parametrization).
Loading checks the method and the layout and restores both states strictly.

## Run

Use the same data directory, test-task count and data seeds as the final runs
of the other methods (model seed 98 is fixed in `model.py`):

```bash
python -u -m cdmetadl.run \
  --seed=93 \
  --input_data_dir=/content/meta_album_final \
  --submission_dir=/content/AG-Meta_Album/baselines/fo_proto_layerlr_maml \
  --output_dir_ingestion=<output>/fo_proto_layerlr_maml_final_data_seed_93/ingestion \
  --output_dir_scoring=<output>/fo_proto_layerlr_maml_final_data_seed_93/scoring \
  --test_tasks_per_dataset=600 \
  --overwrite_previous_results=False \
  --save_train_raw_outputs=False
```

or `run_baseline_experiment("fo_proto_layerlr_maml", ...)` with the options of
the existing final runs. Repeat for data seeds 97 and 101.

## Verification

```bash
python -B -m unittest discover -s baselines/fo_proto_layerlr_maml/tests -v
```

Ten CPU tests on synthetic data: zero-initialized adaptation and encoder
meta-gradients equal to the actual FO-Proto-MAML helper (float32 and float64,
exact); no RNG use and identical ResNet18 initialization; unbounded per-tensor
step and unchanged classifier step; per-step rows; detached support gradients
and the closed-form step-size meta-gradient above; variable way and no episode
state leakage; config and copied-file parity; metrics; Adam update, metrics
file, best-validation snapshot and checkpoint round trip with layout and method
checks; real ResNet18 layout (60 tensors) and query backward.

All ten pass. No Meta-Album training or test was run while preparing this
baseline.

Related but different: `fo_proto_ema_learnedlr_lrsgmaml` learns per-step,
per-tensor rates bounded to [0.001, 0.05] on top of the EMA low-rank transform.
