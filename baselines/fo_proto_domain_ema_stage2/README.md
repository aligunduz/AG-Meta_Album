# FO-Proto Domain EMA Stage2

Learns domain-specific low-rank coefficients on a **frozen shared encoder and
U,V basis**. Starts from the actual `fo_proto_constz_lrsgmaml` EMA warmup
`max-va.pth`; never trains stage 1. No learned-LR features are included.
Requires this repository and the source baseline beside this directory. All
experiment code lives in this directory. Run only `train_stage2.py` and
`eval_stage2.py`; unchanged native ingestion does not pass domain metadata and
does not support this experiment's routing.

## Protocol

- FEEDBACK only: Set-0 split into 7 train / 3 validation datasets using
  `data_seed`; all 10 Set-1 datasets are test. Dataset order is preserved.
- `domains.json` records the supplied taxonomy from [Table 2, page 6](https://meta-album.github.io/paper/Meta-Album.pdf#page=6).
  Order defines stable expert IDs 0–9. Seven are instantiated from the split.
  Supported leaf aliases: exact ID, `ID_Mini`, `DOMAIN_ID.ID`. A set directory
  prefix is allowed. No fuzzy matching or `info.json` domain field is required.
  Folder IDs, metadata/labels hashes, alias collisions and complete split
  membership are checked at runtime. Data are absent in this checkout.
- Default `stage2_seed = 100000 + data_seed`. This seed changes training
  episodes only. Validation/test seeds default to `data_seed`; explicitly set
  them if the source run used different episode seeds.
- Every specialist and the extra-training control are independent copies of
  G0/m0. All new counters start at zero and memories are initialized.
- Default B=1,000 per specialist: 7,000 accepted tasks, **14,000 training
  episode/branch operations** including G_all. One continuous generator call
  skips domains whose quota is full; it never restarts the seed in chunks.
- Every new memory uses alpha=.1 for its first 100 tasks, then .01. Task 1
  also applies EMA to m0. Adaptation uses history; the support mean is committed
  only after query gradients. G_all warms up on its own first 100 tasks.
- Inner steps/LRs, outer LR, clipping, summed gradients and meta-batch come
  from the source checkpoint. Defaults there are 5 inner steps, .01 inner
  LRs, .001 Adam LR, clipping 10, meta-batch 2. Each specialist accumulates its
  own two tasks, even when other domains appear between them.
- Default: each specialist has 500 Adam steps; specialists total 3,500;
  G_all has 3,500. Actual tasks and Adam internal steps are checked.
- For B not divisible by meta-batch, flush the unscaled gradient sum at each
  branch's final budget. The early G_all snapshot is taken after task B without
  flushing its pending batch. It then has floor(B/meta-batch) steps, so only
  divisible budgets give an exact step-matched comparison. Final G_all has
  ceil(7B/meta-batch) steps. No extra step is added at the early snapshot.
- Encoder leaves remain differentiable for inner adaptation, but are excluded
  from optimizers. Outer gradients are requested only for the active GateNet.
  U/V, scalar gates, G0 and m0 are frozen. Source functional BN/preprocessing
  is reused; all shared parameters and buffers are checked for exact equality.

## Training (Colab / repository root)

Run the following command from the repository root in a Colab `%%bash` cell.
Replace the paths below and
use the **source run's data seed**. The data root must already contain
`info/meta_splits.txt` with all Set-0 entries under `meta-train` and Set-1
entries under `meta-test`. No data are downloaded or split lists rewritten.

```bash
python baselines/fo_proto_domain_ema_stage2/train_stage2.py \
  --source_checkpoint /content/STAGE1/model/max-va.pth \
  --input_data_dir /content/FEEDBACK \
  --output_dir /content/RUNS/stage2_seed93 \
  --data_seed 93 --stage2_seed 100093
```

Optional: `--budget_per_domain`, `--validation_seed`, `--test_seed`,
`--image_size`, `--source_provenance /path/provenance.json`, and `--options`
for overrides of `config.json` (including EMA schedule/tolerances/bootstrap).
The output directory must not exist. CLI training runs validation checks,
but does not evaluate test tasks. Output: `stage2-final.pth` and
`stage2-all-stepmatched.pth`; neither is called best-validation.
The checkpoints are written atomically before final budget/frozen-state checks
and after-training validation. `summary.json` records verification success or
failure before any error is propagated. Keep the training `summary.json` beside
`stage2-final.pth`: loading requires a successful report with its matching
checkpoint hash. Failed verification preserves the weights for investigation.

## Paired evaluation

```bash
python baselines/fo_proto_domain_ema_stage2/eval_stage2.py \
  --checkpoint /content/RUNS/stage2_seed93/stage2-final.pth \
  --source_checkpoint /content/STAGE1/model/max-va.pth \
  --input_data_dir /content/FEEDBACK \
  --output_dir /content/RUNS/stage2_seed93_eval \
  --test_tasks_per_dataset 100
```

Optional: `--stage1_results /path/old_tasks.csv`, `--test_seed`.
The checkpoint's image size, split
manifest and source hash must match. Changing test seed is recorded explicitly.
Set task count/seed to the original evaluation values for historical comparison.

One materialized task is used for all conditions:

| Condition | Seen training domains | Unseen domains |
|---|---|---|
| A | G0/m0 | G0/m0 |
| B | matching specialist | G0/m0 |
| C | final G_all/m_all | G0/m0 |
| D | G_all/m_all after B tasks | G0/m0 |

For the seen domains, all other six specialists are evaluated for the matrix.
These measurements never select experts or checkpoints. Each adaptation starts
from the same shared encoder, never another branch's fast weights.

Outputs: `tasks.csv` (sequential `task_id`, separate `episode_hash`, dataset,
ways/shots, branch/expert, loss, ordinary accuracy,
predictions and timing), `test-manifest.jsonl`, `domain-specialist-matrix.csv`,
`matrix-tasks.csv`, and `summary.json`. Summaries include dataset/domain,
seen/unseen/overall results; B−C, B−A, B−D; delta norms and distances from
G0(m0); counters; verification and timing. Primary `accuracy` is the mean of
per-task accuracy, matching benchmark scoring: every task has equal weight
regardless of its query count. B−C, B−A, B−D and the specialist matrix use this
same task mean. Query-micro and dataset-macro accuracy are secondary metrics.
The B−C percentile 95% CI uses 10,000 paired episode bootstrap draws
**within fixed dataset strata**, with separate RNG seed 2026. This CI is
conditional on these datasets and preserves each dataset's task count; equal
counts give equal dataset weights. Extra adaptations for matrices and parity checks
are counted separately from the 14,000 training operations.

## Local data and learner adapters

`sampling.py` owns continuous sampling and wraps each unchanged `Task` in a
local `Episode` with a separate manifest. `data.py` reads the actual
`Task.dataset` to resolve its exact dataset/domain mapping. Neither the shared
Task nor datasets receive new attributes. Shared modules are imported without
monkey patching. The source baseline's legacy sibling imports are resolved
inside private module namespaces without replacing global import entries.

The local sampler preserves the reference dataset/class/image/shuffle RNG
order and uses the existing loader's way/shot configuration logic. Before
validation/test tasks reach a model, their dataset, ways/shots, class order,
ordered images and labels are compared with the unchanged reference generator.
This adds data-loading work, but no model adaptations. A mismatch is an error.
Training uses one continuous local RNG; it does not call the shared generator
with unsupported unbounded arguments or restart it in chunks.

`eval_stage2.py` explicitly passes context derived from `Task.dataset` to
`Stage2Learner.set_task_context`, then calls the unchanged five-item
`fit((x,y,orig_y,ways,shots))` interface. This local adapter is checked against
condition B. Missing context and unknown IDs fail; known unseen domains use
the experiment's explicit G0/m0 policy. No domain is inferred from episode
order, caller variables, class IDs or query labels.

No native ingestion hooks, shared runner flags, notebook changes or external
wrapper integration are provided. Use fresh output directories for the two
script commands above.

## Provenance and prepared checks

The source path/SHA256, current Git revision and code hashes, source config,
taxonomy, ordered split manifest, all seeds and source provenance are saved.
The source baseline's normal checkpoint **does not contain a data seed or
historical episode manifest**. `data_seed_verified=false` explicitly marks a
user-supplied seed without evidence; historical score equality is not claimed.
An optional provenance JSON may provide `data_seed`, `split_manifest` (same
schema as the generated summary), `validation_seed`, `image_size`,
`source_sha256`, `code_sha256`, `validation_manifest_sha256`, and
`accuracy_aggregation: "query_micro"`. Supplied conflicting evidence fails.

Validation uses G0 only and retains the source's query-micro aggregation.
Before/after manifests and predictions must remain unchanged; logits and loss
use numeric tolerances, not bitwise output-hash equality. Output hashes remain
available for auditing. The actual source `MyLearner.load` is the oracle, and per-task logits
and predictions are compared, not just mean accuracy. Historical checkpoint
score equality is enforced only with verified seed/manifest/aggregation.
Otherwise the summary explicitly reports that historical validation cannot be
verified. Default numeric tolerance: `atol=1e-6`, `rtol=1e-5`; argmax predictions
must match exactly. Zero-training deltas are checked exactly for every branch;
each specialist's first task is also checked against the source loader before
training it. Final evaluation checks source A and the local learner adapter's B
predictions per task, and verifies
all GateNet/EMA states and counters remain unchanged.

Episode IDs include dataset, classes, ordered sample indices/files, ways/shots
and hashes of realized support/query tensors and labels. Native stage-1 CSVs
are matched by sequential `task_id`: task i maps to ID i. The comparison checks
dataset, ways, shots and task accuracy. IDs must cover the entire evaluation
without duplicates; CSV structure is checked before model evaluation. Compatible
legacy episode-hash IDs are also supported. Predictions and episode hashes are
checked when supplied. Matching index/metadata/accuracy does not by itself prove
identical sampled images. Same seed alone is not proof.

Evaluation saves its measurements before historical comparison and records any
comparison failure in `summary.json` before raising. Partial reports and task
CSVs are also retained when other evaluation checks fail.

This change was reviewed statically only. Training, inference, installation,
data download and test commands have **not** been run. The checks above execute
when the user runs the experiment. This experiment does not separately identify
EMA/warmup effects and does not establish an upper bound for learned routers.
