# FO-Proto-Global-LRSGMAML

Standalone competition baseline based on `fo_proto_lrsgmaml`, with one global
learnable rank residual vector per low-rank encoder layer:

```text
G = stop_gradient(clipped_support_gradient)
projected = V.T @ G.reshape(out_dim, -1)
projected = (1 + global_delta_c).unsqueeze(1) * projected
correction = beta * (U @ projected)
G_tilde = sigmoid(a) * G + correction.reshape_as(G)
```

`GlobalLowRankTransport.global_delta_c` in `low_rank_transport.py` is an
`nn.ParameterDict` keyed by the same encoder indices as U/V. Each vector has
`min(rank, out_dim)` entries, exactly matching its layer's effective LRSG rank.
The vectors start at exact zero without consuming RNG. Conv/Linear weights
receive the low-rank correction; bias, BN and other 1D tensors retain the
original scalar gate only. No GateNet, task embedding or `condition()` exists.
Support features are still computed for the ordinary prototype classifier.

The unchanged adaptation helper uses sibling encoder clones and prototype W,b,
`create_graph=False`, the same gradient clipping and the same encoder/classifier
learning rates. W,b receive their ordinary updates without transport. All
transport parameters remain outside the fast weights. The outer Adam receives
`encoder_parameters + list(transport.parameters())`, including global_delta_c,
with the same per-task clipping and summed meta-batch gradients as LRSG.

With global_delta_c=0, fast weights and query logits match LRSG exactly, both
at initialization and with nonzero learned U/V. Because U initially equals
zero, V and global_delta_c have connected zero gradients at the first outer
backward. Once U learns, both can receive nonzero outer gradients. No special
initialization or update is introduced to bypass this behavior.

The config is identical to FO-Proto-LRSGMAML except for the method identifier
`fo-proto-global-lrsgmaml`. Rank, beta, scalar initialization, seeds, ResNet18,
inner steps, LRs, clipping, sampling and meta-batch semantics are preserved.
`MyMetaLearner`, `MyLearner` and `MyPredictor` expose the same competition API.
`max-va.pth` saves the best-validation encoder/transport states, including
global_delta_c and its ordered layer/rank layout, and restores them strictly.
As in LRSG, it is an inference checkpoint, without optimizer resume state.

Existing logging/metrics are retained; the reported correction norms and
ratios include the global coefficient effect. `api.py`, `network.py`,
`weight_names.py`, `metadata`, `metrics.py`, `test.py`, and the renamed adaptation
helper are byte-for-byte copies of the LRSG files. Runtime imports are local
to this submission directory; sibling baselines are only read by unit tests.

Use the competition runner with
`--submission_dir=baselines/fo_proto_global_lrsgmaml` and your existing data,
seed and output settings. The copied `test.py` accepts the same checkpoint
and evaluation arguments as LRSG.

CPU/synthetic validation from the repository root:

```bash
python -m unittest discover -s baselines/fo_proto_global_lrsgmaml/tests -v
```

The tests cover exact LRSG equivalence, initialization/RNG, Conv/Linear and
scalar-only behavior, outer gradients and optimizer membership, fast-weight
exclusion, first-order/clipping/head updates, absence of task conditioning,
checkpoint round-trip, disabled transport, variable-way evaluation, and a
ResNet18 synthetic backward. No full Meta-Album training is needed.
