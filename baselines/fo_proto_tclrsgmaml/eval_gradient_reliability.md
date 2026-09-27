# Support sampling and gradient reliability

`eval_gradient_reliability.py` is an observational diagnostic for an existing
LR-only TC `max-va.pth`. It is **not an official benchmark evaluation**. It adds
no outer loss or optimizer and does not edit `config.json` or training code.

## Colab invocation

From the repository root, using an existing checkpoint and prepared data:

```bash
python baselines/fo_proto_tclrsgmaml/eval_gradient_reliability.py \
  --checkpoint /content/ag_meta_outputs/fo_proto_tclrsgmaml/ingestion/model/max-va.pth \
  --input_data_dir /content/meta_album_feedback \
  --data_seed 94 \
  --sampling_seed 12345 \
  --num_tasks 100 \
  --num_repeats 3 \
  --image_size 128 \
  --output_dir /content/gradient_reliability_seed94_sample12345
```

Replace the checkpoint/data paths and **94 with the checkpoint training run's
actual data seed**. The existing checkpoint format does not save the runner's
data seed or image size; these must come from that run's log. `--data_seed` is
required and its provenance is recorded as user supplied. If a saved config
does contain `data_seed` at the top level or under `experiment_config`, the
script rejects a mismatch. Image size defaults to the runner's 128; use the
training value if different. Existing output directories are rejected, even
when empty; choose a new name for another run.

The script loads with `MyLearner.load`, validates the saved method and
`lrsg.enabled=true`, `task_conditioning.enabled=true`,
`scalar_delta_scale=0.0`, `low_rank_delta_scale=1.0`. Encoder/LR/clip settings
come from the saved config. Only the diagnostic's local adaptation config sets
five steps, as requested; the learner config is not changed. The script checks
encoder/transport parameter and buffer equality on exit (including exceptional
exit), config equality and absence of persistent parameter `.grad` values on
normal completion. No optimizer is constructed or stepped.

## Sampling

`prepare_datasets_information(..., validation_datasets, data_seed)` rebuilds the
training run's meta-validation split; its **second** return value is used.
`ImageDataset` supplies the existing Resize/ToTensor preprocessing. No image
augmentation or normalization is added. The custom sampler avoids the normal
episode generator's fallback that can reduce shots or ways on small datasets.

- `--num_tasks=100` means **100 task attempts in total**, not 100 per dataset.
  Each attempt chooses a validation dataset uniformly, then five eligible
  classes without replacement. Counts per dataset are reported.
- Each eligible class needs at least **30 distinct-path images**: 20 query,
  five support A and five support B. All rows whose resolved image path occurs
  more than once in the dataset are excluded to prevent index-only disjointness.
  Missing file paths are excluded before eligibility and counted separately.
- A task samples its classes and query once. Three support pairs are sampled
  from the remaining images. Within each pair A/B/query are mutually disjoint;
  across repeats, supports **may overlap**. The query is fixed across all pairs
  and both evaluation conditions. Separate RNG streams keep task classes/query
  fixed when the repeat count changes.
- Classes with fewer than 30 usable images are excluded and listed with counts
  and reasons. If fewer than five classes remain, the selected dataset's task
  attempt is skipped. Skips are not replaced. Ways, shots and query counts are
  never reduced. The inventory includes datasets even if never sampled.
- Image read/non-RGB failures skip the entire task before any computation on it;
  sampled indices remain in the manifest. Other computation errors terminate
  the run and mark partial results `failed`. No task contributes partial repeats.

Indices are zero-based `ImageDataset` / `labels.csv` row positions; ordered
indices, file paths, local labels, dataset class IDs and dataset label-file
hashes are saved. Paths, rather than image-content hashes, define disjointness.
The manifest records completed/skipped/failed/planned tasks, including reasons.

## First-step measurement

For each support set, the same checkpoint encoder initializes a fresh
prototype head `W=2p`, `b=-||p||²`. Encoder tensors and head tensors are
independent autograd leaves for this measurement. Thus the support derivative
holds the head fixed, matching the normal inner coordinates. There is no
accidental derivative through prototype construction into the encoder.

The first support gradients use the saved elementwise clipping **before**
transport. Let the clipped encoder gradients be `g_A`, `g_B`. Per encoder tensor:

```text
g_mean = (g_A + g_B) / 2
d      = (g_A - g_B) / 2
r0     = ||d||² / ||g_mean||²
rP     = ||P(d)||² / ||P(g_mean)||²
```

The A support's initial embeddings generate coefficients once. With those
coefficients frozen, both `d` and `g_mean` pass through **the same P_A**. The
measurement is repeated with frozen P_B. The existing `transport_gradient`
computes the full scalar gate plus low-rank residual; no dense P is built.
The result includes every encoder tensor, including scalar-only BN parameters.
`rP/r0 > 1` means relative disagreement is larger under that frozen transform
for this pair; it does not by itself establish a loss in useful information.

Separately, total system sensitivity compares `t_A=P_A(g_A)` with
`t_B=P_B(g_B)`:

```text
system_difference_sq   = ||t_A - t_B||²
system_disagreement_sq = ||(t_A - t_B)/2||²
system_mean_sq         = ||(t_A + t_B)/2||²
r_system               = system_disagreement_sq / system_mean_sq
```

These system fields repeat identically in each pair's A/B frozen-P rows. They
are never substituted for `rP`. Differences include changes in both prototype
heads and functional BN batch statistics, as well as support examples; they
are not pure noise estimates for a fixed classifier or fixed BN statistics.

Squared norms accumulate in **float64**. Gradients, mean/difference and the
transport itself retain checkpoint arithmetic. System differences are computed
after casting transformed gradients to float64. The configurable
`--min_norm_sq` (default `1e-20`, squared norm units) invalidates zero or tiny
denominators. There is **no added epsilon**. `rP/r0` also requires raw
disagreement energy above that floor and a positive r0. Nonfinite energies or
ratios are invalid. Numerators/denominators remain separate columns; invalid
values are blank in CSV / `null` in JSON, with explicit status fields.

## Paired five-step benefit

Each A and B support is evaluated separately on the same fixed query under:

1. Normal LR-only TC.
2. Whole low-rank residual off, with learned scalar `sigmoid(logit)` retained.

Both paths use the unchanged `adapt()` helper and the checkpoint's encoder LR,
classifier LR and clipping. A read-only transport view replaces each rank
coefficient delta with `-1` for condition 2, so the existing `(1+delta_c)`
projection multiplier is zero. Neither disabling all transport nor setting
`low_rank_delta_scale=0` would implement this control. No persistent config,
parameters or buffers are edited. Prototype initialization and ordinary
classifier updates are preserved. Conditioning is computed once per support,
as in normal adaptation. Autograd is enabled for inner gradients; transport
coefficients/outputs do not need outer graphs in this diagnostic.

Accuracy is the usual argmax fraction; query cross-entropy uses logits.
`accuracy_gain = accuracy_on - accuracy_off` and
`loss_gain = CE_off - CE_on`: **positive always favors low-rank on**.

## Outputs and aggregation

| File | Contents |
| --- | --- |
| `layer_pairs.csv` | Task, pair, encoder tensor, frozen P_A/P_B, squared energies, ratios and validity statuses |
| `support_benefits.csv` | Both five-step conditions and gains for each pair's A/B support |
| `task_benefits.csv` | One row per completed task, mean across its support pairs and A/B evaluations |
| `task_layer_means.csv` | One row per task/tensor/frozen-P source; mean valid pair ratios and valid-pair counts |
| `sampling_manifest.json` | Eligibility inventory, skip reasons, class selection and all sample indices/paths |
| `metadata.json` | Saved config, architecture, seeds/provenance, protocol, hash identifiers, numerical policy, run status |
| `summary.json` | Dataset means; tensor/frozen-P summaries; within-dataset task-level Pearson correlations |
| `summary.txt` | Short dataset and tensor summaries, task counts and run status |

Ratios are averaged across valid repeats **within each task first**, separately
for P_A and P_B. Dataset/tensor summaries then weight those tasks equally.
Invalid repeats are excluded, not replaced by zero; valid-pair/task counts are
reported. A mean of ratios is not a ratio of pooled energies. Gain estimates
average all six support evaluations per task by default. Correlations relate
task-level `rP/r0` or `r_system` to task-level accuracy/loss gain within each
dataset/tensor/P source, requiring at least three finite task pairs and nonzero
variance. Constant/insufficient groups return null plus reason. Support repeats
are never counted as independent tasks. Correlations are exploratory,
descriptive associations, without significance tests or causal claims.

On failure, output JSON/text marks the run failed; CSVs may contain earlier
completed tasks. A new run must use a different directory. No diagnostic,
training, inference, test or installation command was executed during the
implementation; runtime behavior and numerical parity remain unverified.
