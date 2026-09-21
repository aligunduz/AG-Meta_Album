# FO-Proto-MAML

Independent submission key: `baselines/fo_proto_maml`; checkpoint method:
`fo-proto-maml`. Algorithmic sources are
[Meta-Dataset, section 3 and appendix](https://arxiv.org/abs/1903.03096) and the official
[MAMLLearner](https://github.com/google-research/meta-dataset/blob/main/meta_dataset/learners/optimization_learners.py)
(`proto_maml_fc_weights`, `proto_maml_fc_bias`, `forward_pass`,
`gradient_descent_step`). No tutorial implementation is used.

Each episode averages support embeddings separately for labels `0..way-1`,
including unequal shot counts. It constructs functional `W=2*c`,
`b=-sum(c*c)` tensors and adapts both encoder and head using support cross
entropy. The query loss trains only the persistent encoder. No cosine,
temperature, or feature/prototype normalization is added. Query labels must
use the same episode class mapping as support labels, as supplied by the
existing loader; invalid indices are rejected.

`create_graph=False` stops derivatives of inner gradients. `retain_graph=True`
preserves the prototype initialization graph for outer backward. Encoder fast
weights are sibling clones of the weights used for initialization: the inner
encoder derivative holds the head fixed, while outer differentiation includes
the direct initialization path. Initialization never detaches or registers a
new Parameter. Adaptation explicitly enables gradients inside validation's
`no_grad` context. Every call constructs a fresh head; no episode state is
written into the encoder. `make_encoder` removes the inherited registered
classifier, so the optimizer and checkpoint contain no classifier parameters.

## Controlled protocol

`config.json` extends `baselines/maml/config.json` without changing any existing
field. The baseline there is already first-order. ResNet18, model seed 98,
data seed 93, transforms, data split selection, episode generators, functional
batch normalization, five inner steps, gradient clipping at 10, Adam at 0.001,
and summed meta-batches of two are retained. Encoder and head learning rates
are separately configurable and initially both 0.01. Transport, task gates,
low-rank transport and random classifier reset are explicitly disabled.

Algorithmic fidelity does not imply using Meta-Dataset's tuned hyperparameters:
these settings deliberately preserve this repository's FOMAML comparison.
Its functional BN uses batch statistics separately on support and query, even
in eval mode. Prototype initialization sees support only; the existing
transductive query BN convention is unchanged.

The current checkout has no registry, YAML, global train.py/test.py, or MAML
DataParallel support. The native integration is submission_dir + JSON +
MyMetaLearner/MyLearner/MyPredictor. Existing methods and their pickle loaders
are untouched. This baseline's versioned encoder-only checkpoint is separate
and is not a converter for old MAML checkpoints. Multi-GPU execution is not
added or claimed; existing methods' execution remains unchanged.

Validation uses the same pooled query accuracy and strict-improvement rule as
FOMAML. `save` writes only the selected snapshot to `max-va.pth`, with method,
full submission config and validation score. `load` accepts that file or its
directory and always selects `max-va.pth`. If training ends before its first
scheduled validation, one validation is required instead of silently saving
the last encoder as the best. This does not change the default 30,000-step
protocol. Optimizer resumption is not implemented, as in the original baseline.
Runner flags (data root, seed, image size) remain in the runner's experiment log;
reuse them for standalone testing.

## Colab commands (repository root)

With data prepared at `/content/meta_album_feedback`, the native runner trains,
validates, saves `max-va.pth`, and then tests that checkpoint:

```bash
python -u -m cdmetadl.run \
  --seed=93 \
  --input_data_dir=/content/meta_album_feedback \
  --submission_dir=/content/AG-Meta_Album/baselines/fo_proto_maml \
  --output_dir_ingestion=/content/ag_meta_outputs/fo_proto_maml/ingestion \
  --output_dir_scoring=/content/ag_meta_outputs/fo_proto_maml/scoring \
  --test_tasks_per_dataset=100 \
  --overwrite_previous_results=False \
  --save_train_raw_outputs=False
```

Test the saved best checkpoint separately without training:

```bash
python baselines/fo_proto_maml/test.py \
  --checkpoint=/content/ag_meta_outputs/fo_proto_maml/ingestion/model/max-va.pth \
  --input_data_dir=/content/meta_album_feedback \
  --seed=93 --test_tasks_per_dataset=100
```

The standalone command uses the runner's test sampling defaults and reports
pooled query accuracy; use the native runner for its complete scoring reports.

## Fast verification

```bash
python -m unittest discover -s baselines/fo_proto_maml/tests -v
python -m unittest discover -s baselines/sgmaml/tests -v
python -m unittest discover -s baselines/tcsgmaml/tests -v
python -m unittest discover -s baselines/lrsgmaml/tests -v
git diff --check
```

New checks cover probability equivalence, unbalanced support counts, isolated
prototype gradients, first-order gradients and the exact surviving head
Jacobian, independent inner coordinates, query exclusion, class ordering,
2/7-way episodes, episode isolation, no_grad validation, config parity,
outer optimization, immutable best selection, checkpoint round-trip, original
ProtoNet equivalence, and a one-step real ResNet CPU backward pass. Existing
SGMAML tests also exercise the unchanged original MAML.

Verified locally on CPU: all 10 new tests, all 8 SGMAML tests and all 11
LRSGMAML tests pass. The unchanged TCSGMAML suite reports 10 errors across
13 tests, including its existing 2-D mock-input versus 4-D image validation
mismatch. `compileall`, standalone `test.py --help`, and `git diff --check`
pass. No long training, dataset benchmark, or GPU run was performed.

Files: `model.py` (submission/training/checkpoint), `helpers_fo_proto_maml.py`
(functional algorithm), `config.json`, `test.py` (standalone best test),
`tests/test_fo_proto_maml.py`, and this README. `network.py`, `weight_names.py`,
`api.py`, `metadata` are unmodified local copies from MAML so this submission
does not depend on another baseline being present on the import path.
