# FO Proto LR hybrid EMA warmup

Notebook key: `fo_proto_lr_hybrid_ema_warmup`. Data seed 93; model seed 98.
No experiment is launched by importing the wrapper or opening the notebook.

## Source mapping and comparison

The supplied run names are external experiment labels: neither appears in this
checkout, and their saved run configs/results are not available here. The code
matching `fo_proto_tclrsgmaml_lr_only_tc_feedback_data_seed_93` is
`baselines/fo_proto_tclrsgmaml`: its checked-in config has scalar residual scale
0, low-rank residual scale 1, hidden size 128, no input normalization, rank 4,
beta 1. The warmup implementation matching
`fo_proto_lr_ema_warmup_feedback_data_seed_93` is
`baselines/fo_proto_constz_lrsgmaml` with `constant_condition.init="ema"`.
The separate `fo_proto_ema_learnedlr_lrsgmaml` directory also contains EMA code;
its name alone does not establish it as the source of that run.
These are implementation matches, not verified provenance of external runs.

This standalone submission copies the ConstZ implementation and changes its
conditioning, instrumentation and persistence locally. Both checked-in source
configs agree on the remaining experiment settings: 30,000 tasks, validation
of 300 tasks every 5,000, meta-batch 2, five inner steps, encoder/classifier LR
0.01, Adam outer LR 0.001, clipping 10, 5-way 10-shot train and 5-shot validation.
Existing baselines and `cdmetadl/ingestion/ingestion.py` remain unchanged.
`experiment_ingestion.py` is a baseline-local copy of the ingestion protocol
with actual-seed propagation and post-prediction logging hooks. The existing `cdmetadl.run` CLI selects it only for this hybrid baseline,
then invokes normal scoring. No Colab notebook edits or new arguments are needed.

## Exact computation and learning

1. Reuse the initial support forward of this model's current ResNet18 encoder.
2. Take the mean across support feature vectors and detach it, exactly as in
   LR-only TC. There is no extra embedding normalization. Dimension is 512.
3. Clone the previous EMA buffer. Compute `alpha*z_task+(1-alpha)*z_ema_before`.
   `hybrid.alpha=0.5` is configurable, finite, in [0,1], and not learned.
4. Send this vector to one GateNet (Linear–ReLU–Linear). `input_norm="none"`;
   the saved `gate_input` equals `z_hybrid`. No second model or GateNet is used.
5. Scalar delta scale remains 0; low-rank delta scale remains 1. The saved
   `delta_c_tau` contains rank residuals in `rank_layout` order, excluding the
   unused scalar output coordinates. Low-rank coefficients are `1+delta_c`.

Before initialization, `z_hybrid=z_task`, but GateNet is bypassed and connected
zero residuals preserve the source first-task exception. `gate_input` then
records the candidate vector, and `gate_bypassed=true` states it was not used.
`z_ema_before` stores the actual zero buffer with `ema_initialized=false`.

The encoder is trainable through the prototype/query paths. The conditioning
branch remains detached, as in TC; no parameters are newly frozen. Encoder,
scalar logits, U/V and GateNet stay in the original Adam optimizer. At initial
U/output-layer zeros some gradients are naturally zero; tests use nonzero
learned state to check gradient flow. Support loss/accuracy refers to the final
already-computed inner-step forward, before that step's update.

## EMA ordering, evaluation and checkpoints

Tasks are processed in generator order, including within a meta-batch. Each
uses the EMA left by the preceding completed task. After query backward,
clipping, buffer accumulation and any due optimizer step, commit the detached
**raw** support mean under `no_grad`. Increment the task counter once. Task 1
copies the embedding; tasks 2–5000 use decay 0.9; task 5001 onward uses 0.99.
The schedule in source code matches the requested schedule. `ema_decay` records
the scheduled decay; `effective_update_decay` records the initialization copy
as 0 and evaluation as null.

Validation/test compute fresh support embeddings and mix with the frozen
checkpoint EMA. Query images/labels are never passed to conditioning or EMA.
Test metadata is supplied by the baseline-local runner hook only after prediction. The
normal five-element support-set API is unchanged. Standalone `test.py` uses
that same post-prediction hook. External evaluation runners should call
`learner.record_test_episode(task, probabilities, ordinal)` after prediction
to retain full task metadata; the plain prediction API cannot supply dataset
or query labels. Test loss is NLL reconstructed from returned probabilities
(clamped at 1e-30), rather than a second model forward.

`max-va.pth` saves the selected encoder, transport, EMA, initialized flag,
completed-task counter, selected training step, architecture and complete
hybrid config. Loading rejects architecture/config mismatches.
For training continuation, call `save_training_checkpoint(path, data_state)`
between completed tasks and `load_training_checkpoint(path)` before `meta_fit`.
This restores optimizer, partial meta-batch gradient buffer, best state,
training/validation counters and Python/NumPy/Torch RNG states. The returned
`data_state` must be restored into the caller's episode sampler. The competition
API exposes generator factories rather than sampler state, so the caller must
supply the remaining episode stream; replaying a fresh generator is not an
exact data-order resume. Training checkpoints are trusted local pickle files.

## Running and outputs

In the existing Colab notebook, change only:

```python
BASELINE = "fo_proto_lr_hybrid_ema_warmup"
DATA_SEED = 93  # Other data seeds are propagated into sampling and embedding metadata.
```

The notebook's existing `run_baseline_experiment(...)` still calls
`python -m cdmetadl.run` with its original options. The repository runner
recognizes this submission directory and selects the local hybrid ingestion
module. Other baseline directories retain their existing ingestion path.
`cdmetadl/ingestion/ingestion.py` and the user's Colab notebook are unchanged.
`baseline_experiments.py` remains an optional local convenience wrapper; Colab
does not need to import or use it.

Logs: `<ingestion output>/model/embeddings/`. The existing notebook already
copies and adds the complete `model/` directory to its W&B artifact and Drive
output, so all vectors and metadata are included without a new file-selection
rule. On Drive they are under `<drive run>/model/embeddings/`.
`run.json` describes the trained embedding source, actual seed, run name and
config. `chunk-*.npz` stores float32 vectors plus int64 record IDs; matching
JSONL contains metadata, IDs and array row offsets. Chunks hold at most 64 tasks
by default and flush at validation/phase completion. Each test task flushes
because the local runner reloads the learner per task.
Record IDs continue across checkpoint reloads and distinguish repeated episodes.
No whole run is held in memory; an interrupted process can lose its unflushed
partial chunk. One writer per output directory is supported.

Dataset names and original class IDs are available. The standard Task API has
no separate domain, sample IDs or episode seeds: these are null. Optional
`support_ids`, `query_ids`, `domain`, `domain_id`, `dataset_id`, `episode_seed`
attributes are used when supplied. Fingerprints require real sample IDs and
include dataset and support/query roles. Ordinals do not identify episodes.
Validation rows identify their current training snapshot and validation round.
Test rows identify the selected best snapshot.

Norms, pair distances/cosines, relative task–EMA distance, delta norm and
per-task low-rank correction summaries are in JSONL. Norm denominators <=1e-12
produce null cosine/ratio, with invalid counts; no epsilon hides the invalidity.
Existing low-rank interval metrics retain their source implementation. Interval scalar summaries are emitted to `lrsg_metrics.jsonl` and stdout at
the existing `log_metrics` frequency; full per-task vector coverage is independent
of that frequency. The unchanged Colab notebook logs its standard accuracy/loss
charts and includes these summary files plus full embeddings in the W&B artifact.
Its stdout parser does not forward the new JSON embedding summaries to live W&B
scalar charts. Those extra live charts require a parser change or an explicitly
configured W&B child-process connection; neither is silently introduced here.
Logging uses detached copies and no RNG. Disable with
`embedding_logging.enabled=false` for controlled equivalence checks.

## Offline analysis

```bash
python baselines/fo_proto_lr_hybrid_ema_warmup/analyze_embeddings.py results/hybrid_seed_93/ingestion/model/embeddings --output embedding_analysis.json --seed 98 --max-tasks 512 --max-pairs 10000
```

JSON output contains full-data task-embedding centers, mean norms and RMS
spread by dataset/domain, center L2 matrices, sampled within/between-dataset
L2/cosine distributions, task–EMA distributions by shot and stage, consecutive
EMA movement around warmup, and descriptive Pearson relationships to recorded
loss/accuracy. Phases, runs and evaluation checkpoints stay separate. Training
uses explicit 5,000-task bins split at warmup. Reservoir sampling retains at
most 512 tasks per group. Pair draws are with replacement, excluding self pairs,
up to 10,000 per group pair, using an isolated analysis seed. Invalid cosines
are counted. All original vectors remain available for other analyses.
The encoder changes during training: temporal distances also reflect changing
representation coordinates and do not establish causal effects on accuracy.

## Focused checks

```bash
python -m unittest discover -s baselines/fo_proto_lr_hybrid_ema_warmup/tests -v
```

Synthetic CPU checks cover alpha endpoints, old-EMA order, first-task bypass,
warmup boundary, frozen evaluation, persistence, partial batch state, chunk/row
matching, analysis, logging/RNG equivalence and encoder/U/V/GateNet gradients.
No Meta-Album training or large evaluation is needed.
