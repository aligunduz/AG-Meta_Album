# FO-Proto-MAML + encoder-only MC2

**FO-Proto-MAML + encoder-only MC2; first-order, prototype-initialized classifier, unchanged classifier updates.**

Submission key: `fo_proto_meta_curvature`; checkpoint method: `fo-proto-meta-curvature`.
This is a controlled comparison with this repository's FO-Proto-MAML and EMA/LRTC
baselines. It is not a reproduction of the original full second-order,
classifier-curvature Meta-Curvature experiments.

## Sources and axis convention

- [Meta-Curvature paper](https://arxiv.org/abs/1902.03356).
- [Official project](https://github.com/silverbottlep/meta_curvature).
- [Official maml.py, inspected commit e9ce6546bbf2f10a29d699f0302594829a277251](https://github.com/silverbottlep/maml/blob/e9ce6546bbf2f10a29d699f0302594829a277251/maml.py).

The official `construct_model` initializes rank-one multipliers to ones and
input/output/filter matrices to identity. Its `transform_gradients` uses
TensorFlow HWIO convolution weights and IO linear weights. In that storage,
the output matrix multiplies on the right. We store OIHW / OI weights and define
`Mo = official mc_out.T`, `Mi = official mc_in`, `Mf = official mc_f`.

With `f = h * kw + w`, flatten PyTorch convolution gradients to `(Cout, Cin, K)`:

```text
G_tilde[o,i,f] = sum_{a,b,c} Mo[o,a] Mi[i,b] Mf[f,c] G[a,b,c]
Linear: G_tilde[o,i] = sum_{a,b} Mo[o,a] Mi[i,b] G[a,b]
                    = Mo @ G @ Mi.T
Vector: g_tilde[j] = scale[j] * g[j]
```

Three separate mode products implement convolutions; no combined Kronecker
matrix is created. Matrices are dense, unconstrained, learnable and initialized
to identity. Vector scales are unconstrained and initialized to ones. Parameters
are shared across tasks and inner steps, with separate matrices for each encoder
tensor. There is no low-rank factorization, gate, EMA, task embedding or learned LR.

## Exact encoder coverage

The unchanged `make_encoder` replaces `model.out` with `Identity` before curvature
construction. `EncoderCurvature.names` preserves `encoder.named_parameters()`
order; `architecture()` records every exact name, encoder shape and curvature
shape in the checkpoint. No BN running statistics enter this mapping.

ResNet18 has these 20 convolution weights and 40 BN affine vectors:

| Encoder parameter name (prefix `P = model.features`) | OIHW shape |
| --- | --- |
| `conv.weight` | `(64,3,7,7)` |
| `P.res_block{0,1}.conv{1,2}.weight` | `(64,64,3,3)` |
| `P.res_block2.conv1.weight` | `(128,64,3,3)` |
| `P.res_block2.conv2.weight`, `P.res_block3.conv{1,2}.weight` | `(128,128,3,3)` |
| `P.res_block2.conv3.weight` | `(128,64,1,1)` |
| `P.res_block4.conv1.weight` | `(256,128,3,3)` |
| `P.res_block4.conv2.weight`, `P.res_block5.conv{1,2}.weight` | `(256,256,3,3)` |
| `P.res_block4.conv3.weight` | `(256,128,1,1)` |
| `P.res_block6.conv1.weight` | `(512,256,3,3)` |
| `P.res_block6.conv2.weight`, `P.res_block7.conv{1,2}.weight` | `(512,512,3,3)` |
| `P.res_block6.conv3.weight` | `(512,256,1,1)` |

Every `conv` above has a matching `bn` with `weight` and `bias` of shape `(Cout,)`:
root `bn.{weight,bias}`, blocks `bn1`, `bn2`, plus `bn3` in blocks 2, 4, 6.
Each receives its own scale vector. Conv matrices have shapes `(Cout,Cout)`,
`(Cin,Cin)`, `(kh*kw,kh*kw)`. The encoder currently has no Linear layer; the
implementation supports encoder Linear `(out,in)` weights with two matrices
and its `(out,)` bias with a scale. The task-local classifier is excluded.

## Gradient and checkpoint behavior

The helper starts the task classifier with support prototypes, `W_k=2p_k`,
`b_k=-||p_k||²`. Encoder fast weights are sibling clones. Support gradients use
`create_graph=False`, `retain_graph=True`, and explicit detach; they are clipped
before MC2. Only encoder gradients are transformed. Classifier updates use the
original clipped gradients and classifier LR. Later classifier gradients may
change as the adapted encoder changes, but their update rule is unchanged.

The transformed gradients are not detached. Query backward reaches all curvature
parameters, the shared encoder initialization and the prototype initialization
graph. Only Adam updates curvature. Validation performs no backward or optimizer
step; loaded Learners additionally freeze curvature with `requires_grad_(False)`.
Task adaptation never writes fast weights into persistent modules. Functional BN
continues to use separate support/query batch statistics with local buffers.

Strict improvement of pooled validation query accuracy selects one snapshot
containing both encoder and curvature, including the parameter mapping. The
versioned `max-va.pth` saves/loads that snapshot, config and validation accuracy.
It is not an optimizer-resumption checkpoint or a converter for other methods.

## Preserved settings and first run

`config.json` is copied from `fo_proto_maml` with only the method identifier and
`gradient_transport=true` changed. `network.py`, `weight_names.py`, `api.py` are
unchanged copies. No existing baseline or ingestion/scoring code is modified.

| Setting | Value from existing code/config |
| --- | --- |
| Encoder | ResNet18, pretrained=False; same initialization and RNG order |
| Model seed | 98 |
| First-run data seed | 94, passed to runner (use 94 for comparison runs too) |
| Inner steps; encoder/classifier LR | 5; 0.01 / 0.01 |
| Outer optimizer/LR | Adam / 0.001, encoder and curvature together |
| Meta-batch | 2 tasks, sum gradients, no averaging |
| Clipping | elementwise ±10 on support gradients before MC2; per-task outer gradients before sum |
| Training budget | 30,000 tasks |
| Validation | 300 tasks every 5,000 training tasks; 3 held-out datasets |
| Train episodes | existing 5-way, 10-shot, 20 query images/class |
| Validation episodes | existing 5-way, 5-shot, 20 query images/class |
| Test episodes | unchanged runner: variable 2–20 ways, 1–20 shots, 20 queries/class |
| Image handling | unchanged loader defaults (128 px), normalization and augmentation |

Curvature initialization uses no random draws. The model seed remains in
`model.py`; data seed is a runner option, not a replacement model seed.

In Colab select:

```python
BASELINE = "fo_proto_meta_curvature"
DATA_SEED = 94
MODEL_SEED = 98  # already fixed in the baseline model.py
```

Reuse your `run_baseline_experiment(...)` call with this baseline key and data
seed. Its definition and W&B wrapper are external to this checkout, so their
signature cannot be verified here. The submission uses the unchanged
MetaLearner/Learner/Predictor API and `logger.log` train/validation callbacks;
existing wrapper logging and ordinary accuracy scoring receive the same outputs.
No new W&B dependency or custom accuracy metric is introduced.

The native equivalent, to run yourself from the repository root:

```bash
python -u -m cdmetadl.run \
  --seed=94 \
  --input_data_dir=/content/meta_album_feedback \
  --submission_dir=/content/AG-Meta_Album/baselines/fo_proto_meta_curvature \
  --output_dir_ingestion=/content/ag_meta_outputs/fo_proto_meta_curvature/ingestion \
  --output_dir_scoring=/content/ag_meta_outputs/fo_proto_meta_curvature/scoring \
  --test_tasks_per_dataset=100 \
  --overwrite_previous_results=False \
  --save_train_raw_outputs=False
```

## Prepared verification (not executed)

```bash
python -m unittest discover -s baselines/fo_proto_meta_curvature/tests -v
```

Tests cover fixed-identity adaptation and meta-gradient equivalence to the actual
FO-Proto-MAML helper; nonsymmetric Conv/Linear transpose checks against the
official layout and scalar index equation; vector scaling; detached support
gradients and query-to-curvature gradients; direct encoder and prototype paths;
unchanged classifier updates and clipping order; variable N-way; episode state
isolation; joint best-validation snapshots; checkpoint round-trip; config/backbone
parity; and a real ResNet18 parameter mapping/backward check.

Standalone saved-checkpoint evaluation is available in `test.py` with data seed
94 by default. No training, inference, test or installation command was run
while preparing this baseline.
