# Shared-Embedding FO-Proto-TCSGMAML

This is a controlled combination experiment, not an exact reproduction of
the FO-Proto-MAML paper or this repository's TCSGMAML. The research question is
whether task-conditioned scalar gates derived from FO-Proto-MAML's own support
embeddings improve its encoder adaptation. The only algorithmic addition to
FO-Proto-MAML is gating the encoder's support gradients.

## Episode and gradient paths

Before adaptation, the main meta-learned encoder computes **one support feature
tensor** using its current initialization. `initialize_task` uses that same
tensor in two branches:

* Class means initialize `W = 2*c` and `b = -sum(c*c)`. This branch retains its
  graph, including the direct outer-gradient path from query loss to encoder
  through prototype initialization. Unequal per-class support counts use means.
* `features.mean(dim=0).detach()` provides the global, example-weighted support
  mean to GateNet. Only this conditioning branch detaches; encoder parameters
  receive no gradient through GateNet's input. There is no separate frozen task
  encoder, external checkpoint, query input, domain label or auxiliary signal.

GateNet is the existing TCSGMAML Linear-ReLU-Linear network (hidden size 128),
with zero output-layer weights/biases, `input_norm="none"`, `delta_scale=1.0`,
and one learned shared scalar logit initialized to 4.0 per encoder tensor.
`sigmoid(shared_logits + delta_scale * GateNet(embedding))` is evaluated once
per episode and its graph-connected values are reused in every inner step.
Optional GateNet input-normalization modes retain TCSGMAML's implementation;
the controlled config uses none. Features and prototypes are never normalized.

Support gradients are computed with `create_graph=False, retain_graph=True`.
After elementwise clipping, each encoder gradient is multiplied by its scalar
gate and encoder LR. The dynamic head receives ordinary clipped SGD using its
classifier LR, without gating. GateNet and shared logits are not fast weights
and are never inner-updated. Encoder fast-weight sibling clones keep the head
fixed in the inner partial derivative while preserving both outer paths.

Even with first-order support gradients, gate multiplication remains in the
graph: query loss trains shared logits and GateNet. At zero output-layer
initialization, hidden-layer gradients are initially zero; output-layer and
shared-logit gradients can learn immediately. Once output weights move away
from zero, hidden-layer gradients can also be nonzero. Initial multipliers
are sigmoid(4), approximately 0.982, not exactly 1. This is the retained
TCSGMAML initialization, not an assertion of exact initial FO-Proto-MAML parity.

The outer Adam optimizer, per-task clipping, summed meta-batch buffer, gradient
assignment before `step`, and subsequent reset all cover the same ordered
list: encoder + shared logits + GateNet. No group is cleared before it is
buffered. Non-divisible train-iteration/meta-batch configurations are rejected
instead of silently dropping a partial batch. The default 30,000 / 2 protocol
is unchanged; the generator is expected to yield the requested episode count.

## Protocol, any-way support, and checkpoints

`config.json` copies every FO-Proto-MAML experiment setting and adds only
`gate_init_logit` and `gate_net` from TCSGMAML. Its method key is
`fo-proto-tcsgmaml` and `task_conditioned_gate` is true. The independent
gradient-transport and low-rank options remain false. Model seed 98, data seed
93, ResNet18, five inner steps, both inner LRs 0.01, Adam outer LR 0.001,
meta-batch 2, gradient clip 10, splits, augmentation and validation protocol
are preserved. No `task_encoder` setting exists.

Only encoder tensors determine GateNet output size. Support class indices
must cover `0..way-1`, and query indices must use the same mapping. Each call
creates fresh fast encoder weights and a head sized from support labels,
checked against the task's way count. Thus 2–20-way tasks need no persistent
classifier or changing gate dimensions. Fast weights and head are episode-local.

Validation enables gradients only where adaptation needs them, under the
existing outer `no_grad` evaluation. The inherited functional BN behavior is
unchanged: support and query use their respective batch statistics, including
in eval mode. Support initialization never reads query data.

Pooled validation query accuracy and strict improvement select one coordinated
snapshot: encoder state, shared logits, GateNet state, GateNet architecture,
encoder parameter-name ordering and delta scale. `max-va.pth` also contains
method, version, complete submission config, model arguments and validation
score. No dynamic head is serialized. Loading checks method/version,
architecture and encoder order, then strictly restores all components. Only
`max-va.pth` or its directory is accepted, never `epoch-last.pth`.

This checkout uses the submission directory API, not a global model registry.
No existing baseline file or checkpoint loader is changed. There is no new
DataParallel support, optimizer-resume support, or checkpoint conversion.
Runner seed/data-path/image-size metadata remains in the existing runner and
notebook logs. Standalone evaluation must reuse those settings.

## Colab usage

In the existing notebook select:

```python
BASELINE = "fo_proto_tcsgmaml"
DATA_SEED = 93
MODEL_SEED = 98
```

Its existing run-name expression creates a separate directory such as
`fo_proto_tcsgmaml_feedback_data_seed_93`. Ensure this new baseline directory
has been copied into the Colab checkout; this implementation is not pushed.

From the repository root, the native runner trains, validates, saves and tests
the best checkpoint, with independent output directories:

```bash
python -u -m cdmetadl.run \
  --seed=93 \
  --input_data_dir=/content/meta_album_feedback \
  --submission_dir=baselines/fo_proto_tcsgmaml \
  --output_dir_ingestion=/content/ag_meta_outputs/fo_proto_tcsgmaml_feedback_data_seed_93/ingestion \
  --output_dir_scoring=/content/ag_meta_outputs/fo_proto_tcsgmaml_feedback_data_seed_93/scoring \
  --test_tasks_per_dataset=100 \
  --overwrite_previous_results=False \
  --save_train_raw_outputs=False
```

Evaluate only the saved checkpoint, without training:

```bash
python baselines/fo_proto_tcsgmaml/test.py \
  --checkpoint=/content/ag_meta_outputs/fo_proto_tcsgmaml_feedback_data_seed_93/ingestion/model/max-va.pth \
  --input_data_dir=/content/meta_album_feedback \
  --seed=93 --test_tasks_per_dataset=100
```

The standalone command uses the reference test sampler and reports pooled
query accuracy. The native runner provides complete scoring outputs. Use the
same feedback/final protocol and test counts as the FO-Proto-MAML comparison.

## Tests and verification status

Tests were **written but not executed**, per the request. No model code,
compilation, training, dataset benchmark or package installation was run.
Review is static only; runtime, GPU behavior, memory consumption and accuracy
remain unverified. Retaining initialization graphs costs memory; first-order
means discarding Hessian terms, not discarding all autograd graphs.

For later execution:

```bash
python -m unittest discover -s baselines/fo_proto_tcsgmaml/tests -v
```

The suite covers shared feature identity and single initial forward, both
gradient branches, exact prototype probabilities, first-order gate gradients,
surviving prototype Jacobian, independent inner coordinates, clipping/update
arithmetic, ungated head, once-per-task gate reuse, 2/7/20-way heads,
support-only initialization, episode isolation, config/init parity, all-group
meta-batch accumulation, joint best snapshots and checkpoint round-trips.

Files: `model.py` integrates the submission, all-group optimizer and checkpoints;
`helpers_fo_proto_tcsgmaml.py` implements shared initialization and adaptation;
`task_gates.py` contains GateNet and shared logits; `config.json` defines the
controlled experiment; `tests/test_fo_proto_tcsgmaml.py` defines tests.
`network.py`, `weight_names.py`, `api.py`, `metadata`, and standalone `test.py`
are local copies from FO-Proto-MAML, allowing independent submission loading.
