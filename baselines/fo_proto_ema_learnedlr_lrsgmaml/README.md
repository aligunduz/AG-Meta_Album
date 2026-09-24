# EMA + warm-up with learned encoder inner learning rates

Independent, local submission copied from `fo_proto_constz_lrsgmaml`.
The original baseline and runner files are unchanged.

For encoder tensor j and step s (five steps), learn one scalar logit:

    alpha[s,j] = 0.001 + 0.049 * sigmoid(logits[s,j])
    theta_next[j] = theta[j] - alpha[s,j] * transported_gradient[j]

Logits have shape [5, number of encoder parameter tensors], initialized to
log(9/40), giving alpha=0.01 up to floating-point rounding. Initialization
consumes no RNG. Tensor means in logs are unweighted by tensor size.
The two task-local prototype classifier tensors retain fixed LR 0.01.

Support autograd retains create_graph=False, retain_graph=True and clipping
before transport. Neither the transport output nor alpha is detached. Query
loss therefore trains LR logits and transport while preserving the existing
first-order rule and connected prototype initialization. Logits join the same
Adam (0.001), per-task elementwise clipping (10), summed meta-batch (2).

Unchanged config: 30,000 training episodes, validation every 5,000 on 300 tasks;
train 5-way/10-shot, validation 5-way/5-shot, 20 queries/class, 3 validation
datasets. EMA first task copies the embedding; tasks 2..5000 use alpha=0.1;
afterwards decay=0.99. EMA updates only after a completed training episode.

`max-va.pth` stores the selected encoder, transport/EMA/count, LR logits and
ordered tensor names/shapes/step count/bounds. Load checks the method and
layout and restores all states strictly. Test uses these learned rates.
`lrsg_metrics.jsonl`, stdout and an already-active W&B run include
`inner_lr/min`, `inner_lr/mean`, `inner_lr/max`, and
`inner_lr/step_1_mean` through `inner_lr/step_5_mean` at existing log intervals.

## Files

Experiment-specific: model.py, config.json, learned_inner_lr.py,
helpers_fo_proto_constz_lrsgmaml.py, metrics.py, tests/test_learned_inner_lr.py,
README.md. Verbatim copies: api.py, network.py, weight_names.py, gate_net.py,
low_rank_transport.py, task_transport.py, test.py, metadata.

## Synthetic verification

From repository root:

```bash
python -m unittest discover -s baselines/fo_proto_ema_learnedlr_lrsgmaml/tests -v
```

Tests check float32/float64 initial adaptation and encoder/transport outer
gradient agreement against the original helper, initialized and uninitialized
EMA; nonzero query gradients for every tiny-encoder tensor and step; Adam
updates; bounds and RNG isolation; strict checkpoint/layout and identical
predictions; best-validation selection in four synthetic training episodes;
metrics and preserved config; real ResNet five-step query backward.

The original baseline suite also ran: 35/36 passed. Its existing
`test_reference_gatenet_layout_initialization_and_rng_are_identical` fails on
CRLF versus LF byte comparison of low_rank_transport.py with the sibling TC
baseline. Normalized text is identical. No original file was changed.

## Colab: same seed 94 comparison

Upload this new folder to the same Colab repository checkout. Keep the same
installed dependencies, dataset directory/contents and GPU/runtime as the EMA
seed 94 run. The exact historical Colab command/data path is not stored in
this checkout. For exact matching, duplicate that cell and change only
submission_dir and the two output directories to the paths below; keep its
other options. Runner seed is 94, while the inherited model initialization
seed remains 98 (do not change model.py to 94).

Example using runner defaults (128 px, 100 test tasks/dataset); replace the
input path with the SAME dataset path from that run. If that run overrides
other runner flags, retain those overrides as well.

```python
!python -m cdmetadl.run \
    --seed=94 \
    --input_data_dir=/content/SAME_DATA_DIRECTORY_AS_EMA_SEED94 \
    --submission_dir=baselines/fo_proto_ema_learnedlr_lrsgmaml \
    --output_dir_ingestion=ingestion_ema_learnedlr_seed94 \
    --output_dir_scoring=scoring_ema_learnedlr_seed94
```

Saved model: ingestion_ema_learnedlr_seed94/model/max-va.pth.
No full training was launched locally. No commit, push, PR or remote operation
was performed.
