# Support–query gradient alignment

`eval_gradient_alignment.py` measures initial encoder gradient alignment for an
existing LR-only TC checkpoint, then compares five-step accuracy and query CE
with the low-rank residual on/off. This is a **custom meta-validation diagnostic**,
not an official benchmark result. No new outer loss or optimizer is created.

## Colab

Set the paths and use the **checkpoint training run's actual data seed**:

```python
!python /content/AG-Meta_Album/baselines/fo_proto_tclrsgmaml/eval_gradient_alignment.py \
  --checkpoint /content/ag_meta_outputs/fo_proto_tclrsgmaml/ingestion/model/max-va.pth \
  --input_data_dir /content/meta_album_feedback \
  --data_seed 94 \
  --sampling_seed 12345 \
  --num_tasks 100 \
  --num_repeats 3 \
  --shots 1 5 10 \
  --image_size 128 \
  --output_dir /content/gradient_alignment_seed94_sample12345
```

`--num_tasks` counts total attempts, not tasks per dataset. Existing output
directories are rejected, even if empty. `--shots` accepts distinct integers
from 1 to 10; the underlying support pool always contains ten images per class.
`--min_norm` optionally changes the L2 norm threshold (default `1e-10`).

Old checkpoints do not record the runner's data seed or image size: recover
these from the training log. Seed provenance is recorded as user supplied;
if config contains `data_seed`, mismatches are rejected. `MyLearner.load` and
the existing LR-only validator enforce the saved method and scales 0.0/1.0.
The saved config supplies clipping, encoder/classifier LRs and other settings;
only a local evaluation config selects five inner steps. Training files and
the loaded config are not edited.

## Sampling and measurement

- Uses the existing data preparation function's **meta-validation** return value
  with the training data seed and unchanged Resize/ToTensor preprocessing.
- Samples a validation dataset uniformly, then five eligible classes uniformly.
  Each needs 30 usable images (20 query + 10 support). Missing paths and all rows
  sharing duplicate resolved paths are excluded, as in the reliability script.
  Class counts, exclusions, and skipped task reasons are recorded; skipped
  attempts are not replaced and shots/ways/query counts are never reduced.
- Classes and query are fixed per task. Each repeat draws ten support images
  per class, disjoint from query. Prefixes of these ordered selections provide
  nested 1/5/10-shot sets. Images are loaded once per repeat and subsets reuse
  those tensors. Repeats may share images. Manifest indices are zero-based
  `ImageDataset`/`labels.csv` positions; ordered selections, local class labels,
  paths and subset positions are saved. Disjointness is by resolved path,
  not by file-content hash. Image loading failures skip the whole task.
- Support/query gradients are taken **before any update**, at the same encoder
  values and the same support-prototype classifier. Head and encoder are
  independent leaves; no prototype-initialization derivative enters the encoder.
  Support gradients receive saved elementwise clipping; query gradients do not.
- Functional BN is unchanged: support and query each use their own batch
  statistics with fresh local running-stat buffers, even in eval mode. Query is
  evaluated as one complete batch. Shot effects therefore include changes in
  support batch statistics and prototype estimates.
- `raw` is clipped support gradient; `base` retains the learned scalar gate;
  `full` includes that gate and the support-conditioned low-rank residual.
  Existing `transport_gradient` is used, without constructing dense P. The base
  view sets rank deltas to `-1`, zeroing `(1+delta_c)` while preserving scalars.
  `has_low_rank` identifies tensors assigned a rank residual branch, not whether
  its learned numerical correction happens to be nonzero.
- Norm/dot accumulation is float64. Encoder cosine uses sums over **all encoder
  coordinates**, including tensors whose individual cosine is invalid; it is
  never an average of tensor cosines. Classifier coordinates are excluded.
- Norms at or below `--min_norm` invalidate the corresponding cosine. No epsilon
  is added. Norm values, including tiny/zero norms, remain recorded; nonfinite
  values and undefined cosines/deltas become CSV blanks / JSON nulls. Status
  fields identify why, and summary reports the counts. Finite norm averages
  include tiny values; a finite norm is not necessarily cosine-valid.
- Five-step benefit uses the reliability script's unchanged adaptation/control
  path, starting over from the checkpoint independently for each condition.
  The measured query gradient is never used to adapt weights or coefficients.
  `accuracy_gain = on - off`; `loss_gain = CE_off - CE_on`. Positive favors on.

## Outputs and interpretation

| File | Contents |
| --- | --- |
| `alignment_repeats.csv` | Task/shot/repeat, encoder-wide cosines, deltas, norms, dots, initial CEs, five-step accuracy/CE/gains and validity flags |
| `alignment_tensors.csv` | The same alignment measurements per encoder tensor, shape and low-rank membership |
| `task_shot_means.csv` | Task/shot means and valid repeat counts |
| `sampling_manifest.json` | Eligibility inventory, task states, fixed query, full support pools and nested sample indices |
| `metadata.json` | Saved config, protocol, source/checkpoint/data hashes, seed provenance, BN/numerical policies, run status and state checks |
| `summary.json` | Dataset × shot means/counts, tensor invalid counts, paired shot differences and within-dataset/shot correlations |

Cosine columns are `cos_raw`, `cos_base`, `cos_full`;
`delta_cos_raw = cos_full - cos_raw` and
`delta_cos_base = cos_full - cos_base`. Norms include the query gradient norm;
all three directions' query dots are stored.

Repeats are averaged within each task before dataset means. Undefined values
are omitted with valid-repeat/task counts; never replaced by zero. Each shot
contrast first computes higher-minus-lower on the **same task and repeat** where
both values are valid, averages within task, then across matched tasks. The
summary lists valid task IDs and repeat counts for each contrasted metric.
These paired means may differ from subtracting the unpaired shot summaries.

Pearson correlations of each delta cosine with accuracy/loss gain use matching
valid repeats before task averaging, separately within each dataset and shot.
At least three valid tasks and nonconstant variables are required; otherwise
the correlation is null with a reason. Repeats are not independent tasks.
These are descriptive associations, with no significance or causality claim.

On normal completion, metadata reports parameter/buffer equality, unchanged
config and no persistent parameter gradients. Parameter/buffer comparison also
runs on exceptional exit; a failure records an error and partial outputs are
marked failed. Checks not completed successfully are not reported as passed.

Implementation was not run for training, inference, diagnostics, tests or
installation. Numerical behavior and runtime parity remain unverified.
