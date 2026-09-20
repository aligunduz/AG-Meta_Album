# TCSGMAML - Task-Conditioned Scalar-Gated MAML

An independent copy of the `sgmaml` baseline. SGMAML gives every fast-weight
tensor one static sigmoid gate shared by all tasks. TCSGMAML keeps that shared
logit and adds a per-episode correction predicted from the support set alone:

```
z_i       = mean over x in S_i of f_frozen(x)
delta_i   = h_psi(z_i)
m_ij      = sigmoid(a_j + delta_ij)
theta'_ij = theta_ij - alpha * m_ij * g_ij
```

`f_frozen` is a frozen encoder, `h_psi` is a small MLP (GateNet), `a_j` is the
shared logit and `g_ij` is the clipped first-order support gradient. One scalar
per parameter tensor, never per parameter element, so the gates stay valid when
the classifier changes shape in any-way testing.

## What is learned and what is not

| Component | Inner loop | Outer loop |
| --- | --- | --- |
| Main model initialization | adapted | learned |
| Shared gate logits `a_j` | fixed | learned |
| GateNet `h_psi` | fixed | learned |
| Frozen encoder `f_frozen` | fixed | never touched |

The encoder is excluded from the optimizer, kept in eval mode, and its features
are extracted under `torch.no_grad`. The GateNet and the sigmoid stay in the
graph, so the query loss of meta-training tasks reaches both the GateNet and the
shared logits even though `second_order` is `False`.

Because `a_j` starts at 4.0 and the GateNet output layer starts at exactly zero,
the first update of a TCSGMAML run equals the SGMAML update for every task. Any
later difference is something the query loss learned.

## The frozen encoder checkpoint

`task_encoder.checkpoint` in `config.json` is mandatory and has no default. A
missing, unreadable or architecturally incompatible checkpoint raises; the
baseline never falls back to a random or ImageNet-pretrained encoder.

Two checkpoint forms are accepted:

* a **weight list**, i.e. the list of tensors in `forward_weights` order that
  `Learner.save` writes to `weights.pickle`. This is the normal case, for
  example `ingestion_output/model/weights.pickle` from an earlier run;
* a **state dict** saved with `torch.save`, which additionally carries the
  BatchNorm buffers.

Do **not** point this at `model_state.pickle`. The outer optimizer updates the
weight list, not the parameters of the `meta_learner` module, so
`meta_learner.state_dict()` stays at its random initialization for the whole
run. `model_state.pickle` is therefore a random encoder, which is exactly the
silent failure this baseline refuses to allow.

### Preprocessing

The encoder sees the support images of the episode exactly as the data
generator produces them: 128x128 RGB tensors, the same `image_size` as the run,
with no extra normalization. It must therefore be trained under the same
preprocessing, on meta-train data only. An encoder that has seen the validation
or test classes invalidates the split, and an ImageNet-pretrained encoder makes
the comparison against SGMAML a comparison against extra external data rather
than against the mechanism.

### `task_encoder.forward`

* `fast_weights` (default) extracts features through
  `ResNet.forward_weights(..., embedding=True)`, which normalizes with the
  statistics of the support batch itself. This is how every network in this
  framework is trained.
* `module_eval` extracts features through the module in eval mode, which needs
  real BatchNorm running statistics. Checkpoints trained through
  `forward_weights` never update those buffers, so this mode raises rather than
  silently extracting unnormalized features.

The support set is always passed as one batch and is never chunked, because a
different batch composition yields a different embedding.

## Configuration

`config.json` matches `sgmaml/config.json` exactly, plus two blocks:

```json
"task_encoder": {"checkpoint": null, "forward": "fast_weights",
                 "num_blocks": 18, "img_size": 128},
"gate_net":     {"hidden_size": 128, "input_norm": "none", "delta_scale": 1.0}
```

`input_norm` and `delta_scale` default to the plain design: `Linear(D, 128) ->
ReLU -> Linear(128, P)` on the raw embedding, with the GateNet correction used
as it comes. Two knobs are available for diagnosis and should be treated as
ablations, not as part of the baseline:

* `input_norm`: `l2` or `layernorm` rescale the embedding before the MLP. Raw
  mean features have very different norms across domains.
* `delta_scale`: multiplies the correction. At `a_j = 4.0` the sigmoid
  derivative is about 0.018, so the gradient that reaches the correction is
  damped by roughly a factor of 57 and the conditioning can stay inert.

## Running

Seed 93, the same command used for the other baselines:

```
python -m cdmetadl.run \
    --seed=93 \
    --input_data_dir=../public_data \
    --submission_dir=../baselines/tcsgmaml \
    --output_dir_ingestion=../ingestion_output_tcsgmaml \
    --output_dir_scoring=../scoring_output_tcsgmaml \
    --overwrite_previous_results=True \
    --test_tasks_per_dataset=100
```

Set `task_encoder.checkpoint` first; the run stops during meta-learner
construction otherwise.

## What the checkpoint stores

`Learner.save` writes the SGMAML files plus three more, all taken from the same
validation iteration:

| File | Contents |
| --- | --- |
| `gate_logits.pickle` | shared logits `a_j` |
| `gate_net.pickle` | GateNet architecture, weights and `delta_scale` |
| `task_encoder.pickle` | encoder weights, BatchNorm buffers and provenance |
| `gate_stats.pickle` | gate summaries recorded at each validation |

The encoder travels inside the checkpoint, so meta-testing never depends on the
original checkpoint path still being reachable. A checkpoint missing any of the
first three components raises instead of being completed with default gates or
a fresh GateNet.

## Reading the gate log

Each validation prints one line:

```
[tcsgmaml] gates @ iteration 5000 | tasks 300 | tensors 62 | mean ... |
min ... | max ... | across-task std mean ... | across-task std max ... |
across-task range max ...
```

`across-task std` is the spread of **one** gate over different tasks. The spread
between different tensors is reported separately as `across_tensor_std` in
`gate_stats.pickle` and is not evidence of conditioning: a model whose gates
differ between layers but never move between tasks is SGMAML with extra steps.
If the across-task spread stays at zero, the mechanism never engaged, and the
run says nothing about whether task conditioning helps.

## Comparing against SGMAML

Building the encoder and the GateNet draws from the global torch RNG, which the
framework also uses to initialize the output layer of every validation and test
task. Both are therefore built inside an RNG snapshot that is restored
afterwards, so a TCSGMAML run and an SGMAML run with the same seed see the same
episodes and the same output-layer initializations, and the two can be compared
task by task.

## Tests

```
python -m unittest discover -s baselines/tcsgmaml/tests -v
```

CPU only, seconds to run, no data required.
