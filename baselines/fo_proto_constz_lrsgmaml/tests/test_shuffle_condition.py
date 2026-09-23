"""Synthetic FIFO shuffle controls; no Meta-Album training/data required."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_fo_proto_constz_lrsgmaml import (
    baseline, helpers, ConstantConditionedTransport, ReferenceTransport, Tiny,
)
from low_rank_transport import LowRankTransport


class ShuffleConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(43)
        self.config = baseline.read_config()
        self.config["constant_condition"] = dict(enabled=True, init="shuffle", ema_decay=.99)
        self.encoder = Tiny().double()
        self.weights = list(self.encoder.parameters())
        self.t = ConstantConditionedTransport(self.encoder, self.config)
        self.x = torch.randn(6, 3, dtype=torch.float64)
        self.y = torch.tensor([0, 1, 0, 1, 0, 1])
        self.query = torch.randn_like(self.x)
        with torch.no_grad():
            self.t.u["0"].normal_(0, .1)
            self.t.gate_net.out.weight.normal_(0, .1)
            self.t.gate_net.out.bias.fill_(.3)

    def adapt(self, support=None):
        return helpers.adapt(self.encoder, self.weights, self.x if support is None else support,
                             self.y, self.config["method_config"], 2, self.t,
                             return_task_embedding=True)

    def test_empty_fifo_is_lrsg_equivalent_even_with_nonzero_gatenet(self):
        with patch.object(self.t.gate_net, "forward", wraps=self.t.gate_net.forward) as gate:
            fast, e = self.adapt()
            condition = self.t.condition()
        self.assertEqual(gate.call_count, 0)
        self.assertEqual(condition[0].count_nonzero().item(), 0)
        self.assertTrue(all(c.count_nonzero().item() == 0 for c in condition[1].values()))
        ref = LowRankTransport(self.encoder, self.config["lrsg"])
        ref.load_state_dict({k: self.t.state_dict()[k] for k in ref.state_dict()})
        for name, p in self.encoder.named_parameters():
            g = torch.randn_like(p)
            self.assertTrue(torch.equal(ref.transport_gradient(name, g),
                                        self.t.transport_gradient(name, g, condition)))
        helpers.query_loss(self.encoder, fast, self.query, self.y)[1].backward()
        self.assertEqual(self.t._shuffle_count, 0)
        self.assertFalse(self.t.m_initialized.item())
        for p in self.t.gate_net.parameters():
            self.assertIsNotNone(p.grad)
            self.assertEqual(p.grad.count_nonzero().item(), 0)
        self.t.complete_training_episode(e)
        self.assertEqual(self.t._shuffle_count, 1)
        self.assertTrue(torch.equal(self.t._shuffle_embeddings[0], e))
        self.assertTrue(torch.equal(self.t.m, e))

    def test_current_task_does_not_leak_and_repeated_task_can_sample_different_history(self):
        history = [torch.full((512,), float(i), dtype=torch.float64) for i in range(1, 5)]
        for e in history:
            self.t.complete_training_episode(e)
        seen = []
        handle = self.t.gate_net.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].clone()))
        try:
            # The RNG evolves while exactly the same task and history repeat.
            for _ in range(12):
                _, current = self.adapt()
                self.assertFalse(torch.equal(seen[-1], current))
                self.assertTrue(any(torch.equal(seen[-1], e) for e in history))
            self.assertGreater(len({value[0].item() for value in seen}), 1)
            self.assertEqual(self.t._shuffle_count, 4)
            # Hold the random draw fixed; changing the current support cannot
            # change the input or transport coefficients for this episode.
            state = self.t._shuffle_generator.get_state()
            self.adapt(self.x)
            previous = seen[-1]
            self.t._shuffle_generator.set_state(state)
            self.adapt(self.x + 20)
            self.assertTrue(torch.equal(previous, seen[-1]))
        finally:
            handle.remove()

    def test_fifo_retains_exactly_last_256_detached_embeddings(self):
        source = torch.arange(512, dtype=torch.float64, requires_grad=True)
        self.t.complete_training_episode(source)
        original = source.detach().clone()
        with torch.no_grad():
            source.add_(1000)
        self.assertTrue(torch.equal(self.t._shuffle_embeddings[0], original))
        for i in range(300):
            self.t.complete_training_episode(torch.full((512,), float(i), dtype=torch.float64))
        self.assertEqual(self.t._shuffle_count, 256)
        order = (torch.arange(256) + self.t._shuffle_next) % 256
        actual = self.t._shuffle_embeddings[order]
        expected = torch.arange(44, 300, dtype=torch.float64).unsqueeze(1).expand(256, 512)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(self.t._shuffle_embeddings.requires_grad)
        self.assertIsNone(self.t._shuffle_embeddings.grad_fn)
        self.assertNotIn("_shuffle_embeddings", self.t.state_dict())
        self.assertNotIn("_shuffle_embeddings", dict(self.t.named_parameters()))
        self.t.float()
        self.assertEqual(self.t._shuffle_embeddings.dtype, torch.float32)

    def test_sampling_is_reproducible_without_consuming_global_rng(self):
        other = ConstantConditionedTransport(self.encoder, self.config)
        other.load_state_dict(self.t.state_dict())
        for i in range(5):
            e = torch.full_like(self.t.m, float(i))
            self.t.complete_training_episode(e)
            other.complete_training_episode(e)
        before = torch.random.get_rng_state().clone()
        for _ in range(10):
            a, b = self.t.condition(), other.condition()
            self.assertTrue(torch.equal(a[0], b[0]))
            for key in a[1]:
                self.assertTrue(torch.equal(a[1][key], b[1][key]))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_eval_uses_fixed_ema_without_sampling_or_mutating_any_history(self):
        for e in (torch.ones_like(self.t.m), torch.full_like(self.t.m, 3)):
            self.t.complete_training_episode(e)
        self.t.eval()
        before = {k: v.clone() for k, v in self.t.named_buffers()}
        position = (self.t._shuffle_count, self.t._shuffle_next)
        rng = self.t._shuffle_generator.get_state().clone()
        seen = []
        handle = self.t.gate_net.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].clone()))
        try:
            for shift in (0, 10, -5):
                fast, e = self.adapt(self.x + shift)
                helpers.query_loss(self.encoder, fast, self.query, self.y)
                self.t.complete_training_episode(e)
                self.t.update_ema(e)
            self.assertTrue(all(torch.equal(value, before["m"]) for value in seen))
        finally:
            handle.remove()
        for name, value in self.t.named_buffers():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertEqual(position, (self.t._shuffle_count, self.t._shuffle_next))
        self.assertTrue(torch.equal(rng, self.t._shuffle_generator.get_state()))

    def test_gatenet_layout_formula_and_outer_gradients_match_tc_lr(self):
        self.t.complete_training_episode(torch.randn_like(self.t.m))
        ref = ReferenceTransport(self.encoder, self.config)
        ref.load_state_dict({k: self.t.state_dict()[k] for k in ref.state_dict()})
        self.assertEqual(self.t.gate_net.config(), ref.gate_net.config())
        self.assertEqual(self.t.rank_layout, ref.rank_layout)
        actual = self.t.condition()
        expected = ref.condition(self.t._shuffle_embeddings[0])
        self.assertTrue(torch.equal(actual[0], expected[0]))
        for name, p in self.encoder.named_parameters():
            g = torch.randn_like(p)
            self.assertTrue(torch.equal(self.t.transport_gradient(name, g, actual),
                                        ref.transport_gradient(name, g, expected)))
        fast, current = self.adapt()
        fifo_before = self.t._shuffle_embeddings.clone()
        helpers.query_loss(self.encoder, fast, self.query, self.y)[1].backward()
        self.assertTrue(torch.equal(self.t._shuffle_embeddings, fifo_before))
        for p in self.weights + list(self.t.parameters()):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        n = len(self.t.names)
        self.assertEqual(self.t.gate_net.out.weight.grad[:n].count_nonzero().item(), 0)
        self.assertGreater(self.t.gate_net.out.weight.grad[n:].norm().item(), 0)
        self.assertIsNone(self.t.m.grad)
        self.assertIsNone(self.t._shuffle_embeddings.grad)

    def test_training_order_and_checkpoint_has_only_inference_condition_state(self):
        config = copy.deepcopy(self.config)
        config["experiment_config"] = dict(train_iterations=4, validation_tasks=2, validate_every=2)
        x, query = self.x.float(), self.query.float()
        tasks = [SimpleNamespace(num_ways=2, support_set=(x + i, self.y, self.y),
                                  query_set=(query - i, self.y, self.y)) for i in range(4)]
        valid = SimpleNamespace(num_ways=2, support_set=(x + 20, self.y, self.y),
                                query_set=(query + 20, self.y, self.y))
        with patch.object(baseline, "ResNet", Tiny), patch.object(baseline, "read_config", return_value=config), patch.object(torch.cuda, "is_available", return_value=False):
            meta = baseline.MyMetaLearner(2, 2, SimpleNamespace(log=lambda *a, **kw: None))
            t = meta.transport
            with torch.no_grad():
                t.gate_net.out.weight.normal_(0, .1)
                t.u["0"].normal_(0, .1)
            params = [p for group in meta.optimizer.param_groups for p in group["params"]]
            self.assertFalse({id(p) for p in params} & {id(b) for b in t.buffers()})
            committed, events, valid_inputs = [], [], []
            forward = meta.meta_learner.forward_weights
            query_loss = baseline.query_loss
            complete = t.complete_training_episode
            def embed(*args, **kwargs):
                result = forward(*args, **kwargs)
                if kwargs.get("embedding") and t.training:
                    events.append("embedding")
                return result
            def gate(module, args):
                if t.training:
                    self.assertTrue(any(torch.equal(args[0], e) for e in committed))
                    self.assertEqual(t._shuffle_count, len(committed))
                    events.append("condition")
                else:
                    self.assertTrue(torch.equal(args[0], t.m))
                    valid_inputs.append(args[0].clone())
            def query_call(*args, **kwargs):
                result = query_loss(*args, **kwargs)
                if t.training:
                    events.append("query")
                    result[1].register_hook(lambda g: events.append("backward"))
                return result
            def publish(e):
                self.assertEqual(events[-1], "backward")
                self.assertEqual(t._shuffle_count, len(committed))
                previous_m = t.m.clone()
                expected_m = previous_m.mul(.99).add_(e, alpha=1 - .99) if committed else e
                complete(e)
                self.assertTrue(torch.equal(t.m, expected_m))
                committed.append(e.clone())
                self.assertTrue(torch.equal(t._shuffle_embeddings[len(committed) - 1], e))
                events.append("publish")
            handle = t.gate_net.register_forward_pre_hook(gate)
            try:
                with patch.object(meta.meta_learner, "forward_weights", side_effect=embed), patch.object(
                        baseline, "query_loss", side_effect=query_call), patch.object(
                        t, "complete_training_episode", side_effect=publish):
                    learner = meta.meta_fit(lambda n: iter(tasks[:n]), lambda n: iter([valid] * n))
            finally:
                handle.remove()
            self.assertEqual(events, ["embedding", "query", "backward", "publish"] +
                             ["embedding", "condition", "query", "backward", "publish"] * 3)
            self.assertEqual(len(valid_inputs), 4)
            self.assertTrue(torch.equal(valid_inputs[0], valid_inputs[1]))
            self.assertTrue(torch.equal(valid_inputs[2], valid_inputs[3]))
            self.assertEqual(t._shuffle_count, 4)
            self.assertEqual(learner.transport._shuffle_count, 0)
            self.assertTrue(torch.equal(learner.transport.m, meta.best_state["lrsg"]["m"]))
            support = (x, self.y, self.y, 2, 3)
            prediction = learner.fit(support).predict(query)
            with tempfile.TemporaryDirectory() as directory:
                learner.save(directory)
                restored = baseline.MyLearner()
                restored.load(directory)
                self.assertEqual(restored.transport._shuffle_count, 0)
                self.assertTrue(torch.equal(restored.transport.m, learner.transport.m))
                rng = restored.transport._shuffle_generator.get_state().clone()
                actual = restored.fit(support).predict(query)
                self.assertTrue(torch.equal(torch.tensor(actual), torch.tensor(prediction)))
                self.assertTrue(torch.equal(rng, restored.transport._shuffle_generator.get_state()))
                data = torch.load(Path(directory) / "max-va.pth", weights_only=True)
                keys = set(data["state"]["lrsg"])
                ordinary_keys = set(ReferenceTransport(self.encoder, self.config).state_dict())
                self.assertEqual(keys - ordinary_keys, {"m", "m_initialized"})

    def test_failed_query_does_not_publish_current_embedding(self):
        config = copy.deepcopy(self.config)
        config["experiment_config"] = dict(train_iterations=1, validation_tasks=1, validate_every=1)
        task = SimpleNamespace(num_ways=2, support_set=(self.x.float(), self.y, self.y),
                               query_set=(self.query.float(), self.y, self.y))
        generator = lambda n: iter([task] * n)
        with patch.object(baseline, "ResNet", Tiny), patch.object(baseline, "read_config", return_value=config), patch.object(torch.cuda, "is_available", return_value=False):
            meta = baseline.MyMetaLearner(2, 2, SimpleNamespace(log=lambda *a, **kw: None))
            with patch.object(baseline, "query_loss", side_effect=RuntimeError("synthetic failed query")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failed query"):
                    meta.meta_fit(generator, generator)
            self.assertEqual(meta.transport._shuffle_count, 0)
            self.assertFalse(meta.transport.m_initialized.item())


if __name__ == "__main__":
    unittest.main()
