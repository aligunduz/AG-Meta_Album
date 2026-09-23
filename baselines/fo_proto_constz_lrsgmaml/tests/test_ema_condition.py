"""Past-episode EMA ordering, evaluation isolation and persistence checks."""
import copy
import contextlib
import io
import json
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
from metrics import log_metrics


class EMAConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(31)
        self.config = baseline.read_config()
        self.config["constant_condition"] = dict(enabled=True, init="ema", ema_decay=.99)
        self.encoder = Tiny().double()
        self.weights = list(self.encoder.parameters())
        self.t = ConstantConditionedTransport(self.encoder, self.config)
        self.x = torch.randn(6, 3, dtype=torch.float64)
        self.y = torch.tensor([0, 1, 0, 1, 0, 1])
        self.query = torch.randn_like(self.x)
        # Nonzero state makes accidental task leakage / first-task GateNet
        # invocation observable, unlike the default all-zero output layer.
        with torch.no_grad():
            self.t.u["0"].normal_(0, .1)
            self.t.gate_net.out.weight.normal_(0, .1)
            self.t.gate_net.out.bias.fill_(.3)

    def adapt(self, support=None):
        return helpers.adapt(self.encoder, self.weights, self.x if support is None else support,
                             self.y, self.config["method_config"], 2, self.t,
                             return_task_embedding=True)

    def test_first_episode_bypasses_gatenet_and_initializes_only_after_query(self):
        before = copy.deepcopy(self.t.state_dict())
        with patch.object(self.t.gate_net, "forward", wraps=self.t.gate_net.forward) as gate:
            fast, embedding = self.adapt()
            a, c = self.t.condition()
            self.assertEqual(gate.call_count, 0)
        self.assertEqual(a.count_nonzero().item(), 0)
        self.assertTrue(all(v.count_nonzero().item() == 0 for v in c.values()))
        # LRSG equivalence must hold even if U and GateNet biases are nonzero.
        lrsg = LowRankTransport(self.encoder, self.config["lrsg"])
        lrsg.load_state_dict({k: v for k, v in self.t.state_dict().items()
                              if k not in ("m", "m_initialized") and not k.startswith("gate_net.")})
        for name, weight in self.encoder.named_parameters():
            g = torch.randn_like(weight)
            self.assertTrue(torch.equal(self.t.transport_gradient(name, g, (a, c)),
                                        lrsg.transport_gradient(name, g)))
        helpers.query_loss(self.encoder, fast, self.query, self.y)[1].backward()
        for p in self.t.gate_net.parameters():
            self.assertIsNotNone(p.grad)
            self.assertEqual(p.grad.count_nonzero().item(), 0)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.t.state_dict()[key]))
        self.assertFalse(embedding.requires_grad)
        expected = self.encoder.forward_weights(self.x, self.weights, embedding=True).mean(0).detach()
        self.t.update_ema(embedding)
        self.assertTrue(self.t.m_initialized.item())
        self.assertTrue(torch.equal(self.t.m, expected))

    def test_current_support_cannot_change_current_condition_or_transport(self):
        previous = torch.randn(512, dtype=torch.float64)
        self.t.update_ema(previous)
        seen, conditions = [], []
        original = self.t.condition
        def condition():
            result = original()
            conditions.append(result)
            return result
        handle = self.t.gate_net.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        try:
            with patch.object(self.t, "condition", side_effect=condition):
                _, e1 = self.adapt(self.x)
                _, e2 = self.adapt(self.x + 9)
        finally:
            handle.remove()
        self.assertFalse(torch.equal(e1, e2))
        self.assertEqual(len(seen), 2)
        for value in seen:
            self.assertTrue(torch.equal(value, previous))
            self.assertFalse(value.requires_grad)
            self.assertNotEqual(value.data_ptr(), self.t.m.data_ptr())
        for name, p in self.encoder.named_parameters():
            g = torch.randn_like(p)
            self.assertTrue(torch.equal(self.t.transport_gradient(name, g, conditions[0]),
                                        self.t.transport_gradient(name, g, conditions[1])))
        self.assertTrue(torch.equal(self.t.m, previous))

    def test_ema_is_detached_buffer_and_does_not_invalidate_pending_backward(self):
        source = torch.randn(512, dtype=torch.float64, requires_grad=True)
        self.t.update_ema(source)
        self.assertIn("m", dict(self.t.named_buffers()))
        self.assertIn("m_initialized", dict(self.t.named_buffers()))
        self.assertNotIn("m", dict(self.t.named_parameters()))
        self.assertFalse(hasattr(self.t, "z"))
        fast, embedding = self.adapt()
        # Even a caller committing before backward cannot mutate the saved
        # GateNet input, because condition() snapshots the buffer.
        before = self.t.m.clone()
        self.t.update_ema(embedding)
        expected = before.mul(.99).add_(embedding, alpha=.01)
        torch.testing.assert_close(self.t.m, expected, rtol=0, atol=1e-15)
        helpers.query_loss(self.encoder, fast, self.query, self.y)[1].backward()
        self.assertFalse(self.t.m.requires_grad)
        self.assertIsNone(self.t.m.grad_fn)
        self.assertIsNone(self.t.m.grad)
        self.assertIsNone(source.grad)
        self.assertGreater(self.t.gate_net.out.weight.grad.norm().item(), 0)

    def test_eval_never_initializes_or_updates_ema(self):
        for initialized in (False, True):
            self.t.train()
            if initialized:
                self.t.update_ema(torch.randn_like(self.t.m))
            self.t.eval()
            before = copy.deepcopy(self.t.state_dict())
            for shift in (0, 5):
                with torch.no_grad():
                    fast, embedding = self.adapt(self.x + shift)
                    helpers.query_loss(self.encoder, fast, self.query, self.y)
                self.t.update_ema(embedding)
            for name, value in before.items():
                self.assertTrue(torch.equal(value, self.t.state_dict()[name]), name)

    def test_ema_gatenet_layout_and_formula_match_tc_lr(self):
        self.t.update_ema(torch.randn_like(self.t.m))
        ref = ReferenceTransport(self.encoder, self.config)
        ref.load_state_dict({k: v for k, v in self.t.state_dict().items()
                             if k not in ("m", "m_initialized")})
        self.assertEqual(ref.gate_net.config(), self.t.gate_net.config())
        self.assertEqual(ref.rank_layout, self.t.rank_layout)
        architecture = self.t.architecture()
        architecture.pop("constant_condition")
        self.assertEqual(architecture, ref.architecture())
        condition = self.t.condition()
        reference = ref.condition(self.t.m)
        self.assertTrue(torch.equal(condition[0], reference[0]))
        for name, p in self.encoder.named_parameters():
            g = torch.randn_like(p)
            self.assertTrue(torch.equal(self.t.transport_gradient(name, g, condition),
                                        ref.transport_gradient(name, g, reference)))

    def test_invalid_decay_rejected_and_zero_needs_no_decay(self):
        for decay in (None, -0.1, 1.0, float("nan"), float("inf"), True, "0.99"):
            config = copy.deepcopy(self.config)
            config["constant_condition"]["ema_decay"] = decay
            with self.assertRaisesRegex(ValueError, "ema_decay"):
                baseline.validate_config(config)
        config = copy.deepcopy(self.config)
        config["constant_condition"] = dict(enabled=True, init="zero")
        baseline.validate_config(config)
        t = ConstantConditionedTransport(self.encoder, config)
        self.assertNotIn("m", t.state_dict())
        self.assertEqual(t.architecture()["constant_condition"],
                         dict(enabled=True, init="zero", shape=[512]))

    def test_meta_fit_episode_order_validation_and_checkpoint_roundtrip(self):
        config = copy.deepcopy(self.config)
        config["experiment_config"] = dict(train_iterations=4, validation_tasks=1, validate_every=2)
        x, query = self.x.float(), self.query.float()
        tasks = [SimpleNamespace(num_ways=2, support_set=(x + i, self.y, self.y),
                                  query_set=(query - i, self.y, self.y)) for i in range(4)]
        valid_task = SimpleNamespace(num_ways=2, support_set=(x + 30, self.y, self.y),
                                    query_set=(query + 30, self.y, self.y))
        with patch.object(baseline, "ResNet", Tiny), patch.object(baseline, "read_config", return_value=config), patch.object(torch.cuda, "is_available", return_value=False):
            meta = baseline.MyMetaLearner(2, 2, SimpleNamespace(log=lambda *a, **kw: None))
            t = meta.transport
            optimized = [p for group in meta.optimizer.param_groups for p in group["params"]]
            self.assertFalse(any(p is t.m or p is t.m_initialized for p in optimized))
            events, inputs, committed = [], [], []
            expected = t.m.clone()
            embed = meta.meta_learner.forward_weights
            query_loss = baseline.query_loss
            update = t.update_ema
            def forward(x, weights, embedding=False):
                result = embed(x, weights, embedding=embedding)
                if embedding and t.training:
                    events.append("embedding")
                return result
            def gate(module, args):
                self.assertTrue(torch.equal(args[0], expected))
                inputs.append((t.training, args[0].clone()))
                if t.training:
                    events.append("condition")
            def query_call(*args, **kwargs):
                result = query_loss(*args, **kwargs)
                if t.training:
                    events.append("query")
                    result[1].register_hook(lambda g: events.append("backward"))
                return result
            def commit(e):
                nonlocal expected
                self.assertEqual(events[-1], "backward")
                self.assertTrue(torch.equal(t.m, expected))
                expected = (expected * .99).add_(e, alpha=1 - .99) if committed else e.clone()
                update(e)
                self.assertTrue(torch.equal(t.m, expected))
                committed.append(t.m.clone())
                events.append("update")
            handle = t.gate_net.register_forward_pre_hook(gate)
            try:
                with patch.object(meta.meta_learner, "forward_weights", side_effect=forward), patch.object(
                        baseline, "query_loss", side_effect=query_call), patch.object(t, "update_ema", side_effect=commit):
                    learner = meta.meta_fit(lambda n: iter(tasks[:n]), lambda n: iter([valid_task] * n))
            finally:
                handle.remove()
            self.assertEqual(events, ["embedding", "query", "backward", "update"] +
                             ["embedding", "condition", "query", "backward", "update"] * 3)
            train_inputs = [value for training, value in inputs if training]
            self.assertEqual(len(train_inputs), 3)
            for value, previous in zip(train_inputs, committed[:-1]):
                self.assertTrue(torch.equal(value, previous))
            valid_inputs = [value for training, value in inputs if not training]
            for value, previous in zip(valid_inputs, committed[1::2]):
                self.assertTrue(torch.equal(value, previous))
            self.assertEqual(len(valid_inputs), 2)
            self.assertTrue(torch.equal(t.m, committed[-1]))
            # Returned learner uses the selected training snapshot; evaluation
            # must consume exactly that checkpoint's EMA without adaptation.
            saved_m = learner.transport.m.clone()
            self.assertTrue(torch.equal(saved_m, meta.best_state["lrsg"]["m"]))
            test_seen = []
            handle = learner.transport.gate_net.register_forward_pre_hook(
                lambda module, args: test_seen.append(args[0].clone()))
            try:
                prediction = learner.fit((x, self.y, self.y, 2, 3)).predict(query)
            finally:
                handle.remove()
            self.assertTrue(torch.equal(test_seen[0], saved_m))
            self.assertTrue(torch.equal(learner.transport.m, saved_m))
            with tempfile.TemporaryDirectory() as directory:
                learner.save(directory)
                restored = baseline.MyLearner()
                restored.load(directory)
                self.assertTrue(torch.equal(restored.transport.m, saved_m))
                self.assertTrue(restored.transport.m_initialized.item())
                actual = restored.fit((x, self.y, self.y, 2, 3)).predict(query)
                self.assertTrue(torch.equal(torch.tensor(actual), torch.tensor(prediction)))
                self.assertTrue(torch.equal(restored.transport.m, saved_m))
                data = torch.load(Path(directory) / "max-va.pth", weights_only=True)
                self.assertIn("m_initialized", data["state"]["lrsg"])
                self.assertNotIn("z", data["state"]["lrsg"])


class EMAWarmupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(31)
        self.config = baseline.read_config()
        self.config["constant_condition"] = dict(
            enabled=True, init="ema", ema_decay=.99,
            ema_warmup_alpha=.1, ema_warmup_tasks=5000)
        self.encoder = Tiny().double()

    def transport(self, config=None):
        return ConstantConditionedTransport(self.encoder, self.config if config is None else config)

    def test_exact_5000_task_schedule_and_first_task_copy(self):
        t = self.transport()
        expected = torch.zeros_like(t.m)
        for task in range(1, 5003):
            embedding = torch.full_like(t.m, float(task % 7 - 3), requires_grad=True)
            if task == 1:
                expected.copy_(embedding.detach())
            elif task <= 5000:
                expected.mul_(.9).add_(embedding.detach(), alpha=.1)
            else:
                expected.mul_(.99).add_(embedding.detach(), alpha=1 - .99)
            t.complete_training_episode(embedding)
            self.assertTrue(torch.equal(t.m, expected), f"task {task}")
            self.assertEqual(t.ema_completed_tasks.item(), task)
            if task == 1:
                self.assertTrue(torch.equal(t.m, embedding))
                self.assertFalse(torch.equal(t.m, .1 * embedding))
        self.assertFalse(t.m.requires_grad)
        self.assertIsNone(t.m.grad_fn)
        self.assertNotIn("ema_completed_tasks", dict(t.named_parameters()))

    def test_warmup_endpoints_one_task_and_alpha_one(self):
        for tasks in (1, 2):
            config = copy.deepcopy(self.config)
            config["constant_condition"].update(ema_warmup_alpha=1, ema_warmup_tasks=tasks)
            t = self.transport(config)
            first, second, third = (torch.full_like(t.m, v) for v in (2., 8., -4.))
            t.complete_training_episode(first)
            self.assertTrue(torch.equal(t.m, first))
            t.complete_training_episode(second)
            expected = second.clone() if tasks == 2 else first.mul(.99).add_(second, alpha=1 - .99)
            self.assertTrue(torch.equal(t.m, expected))
            t.complete_training_episode(third)
            expected.mul_(.99).add_(third, alpha=1 - .99)
            self.assertTrue(torch.equal(t.m, expected))

    def test_absent_fields_preserve_legacy_arithmetic_and_checkpoint_layout(self):
        config = copy.deepcopy(self.config)
        config["constant_condition"] = dict(enabled=True, init="ema", ema_decay=.99)
        for dtype in (torch.float32, torch.float64):
            t = self.transport(config).to(dtype=dtype)
            expected = torch.zeros_like(t.m)
            for task in range(30):
                e = torch.randn_like(t.m)
                if task == 0:
                    expected.copy_(e)
                else:
                    # Exact pre-warmup implementation, including operation order.
                    expected.mul_(.99).add_(e, alpha=1 - .99)
                t.complete_training_episode(e)
                self.assertTrue(torch.equal(t.m, expected))
            self.assertEqual(t.architecture()["constant_condition"],
                             dict(enabled=True, init="ema", shape=[512], ema_decay=.99))
            reference_keys = set(ReferenceTransport(self.encoder, config).state_dict())
            self.assertEqual(set(t.state_dict()) - reference_keys, {"m", "m_initialized"})
            self.assertFalse(hasattr(t, "ema_completed_tasks"))

    def test_validation_of_warmup_fields(self):
        for missing in ("ema_warmup_alpha", "ema_warmup_tasks"):
            config = copy.deepcopy(self.config)
            del config["constant_condition"][missing]
            with self.assertRaisesRegex(ValueError, "appear together"):
                baseline.validate_config(config)
        for mode in ("zero", "shuffle"):
            config = copy.deepcopy(self.config)
            config["constant_condition"]["init"] = mode
            with self.assertRaisesRegex(ValueError, "only valid"):
                baseline.validate_config(config)
        for field, bad_values in (
                ("ema_warmup_alpha", (0, -1, 1.01, float("nan"), float("inf"), True, None, ".1")),
                ("ema_warmup_tasks", (0, -1, 2.0, True, None, "5000"))):
            for value in bad_values:
                with self.subTest(field=field, value=value):
                    config = copy.deepcopy(self.config)
                    config["constant_condition"][field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        baseline.validate_config(config)
        baseline.validate_config(self.config)

    def test_eval_never_changes_ema_or_schedule_count(self):
        for initialized in (False, True):
            t = self.transport()
            if initialized:
                t.complete_training_episode(torch.ones_like(t.m))
            t.eval()
            before = copy.deepcopy(t.state_dict())
            t.condition()
            t.complete_training_episode(torch.full_like(t.m, 100))
            t.update_ema(torch.full_like(t.m, -100))
            for name, value in before.items():
                self.assertTrue(torch.equal(t.state_dict()[name], value), name)

    def test_checkpoint_persists_count_and_rejects_different_schedule(self):
        config = copy.deepcopy(self.config)
        config["constant_condition"]["ema_warmup_tasks"] = 3
        t = self.transport(config)
        t.complete_training_episode(torch.ones_like(t.m))
        t.complete_training_episode(torch.full_like(t.m, 3))
        state = baseline.snapshot(self.encoder, t)
        args = dict(num_classes=2, dev=torch.device("cpu"), num_blocks=18, pretrained=False)
        with patch.object(baseline, "make_encoder", side_effect=lambda args: Tiny().double()), patch.object(
                torch.cuda, "is_available", return_value=False), tempfile.TemporaryDirectory() as directory:
            learner = baseline.MyLearner(args, state, config, .5)
            learner.save(directory)
            restored = baseline.MyLearner()
            restored.load(directory)
            self.assertEqual(restored.transport.ema_completed_tasks.item(), 2)
            self.assertTrue(torch.equal(restored.transport.m, t.m))
            restored.transport.train()
            for value in (-2., 8.):
                e = torch.full_like(t.m, value)
                t.complete_training_episode(e)
                restored.transport.complete_training_episode(e)
                self.assertTrue(torch.equal(restored.transport.m, t.m))
                self.assertEqual(restored.transport.ema_completed_tasks.item(), t.ema_completed_tasks.item())
            original = torch.load(Path(directory) / "max-va.pth", weights_only=True)
            for change in (dict(ema_warmup_alpha=.2), dict(ema_warmup_tasks=4), None):
                data = copy.deepcopy(original)
                constant = data["config"]["constant_condition"]
                if change is None:
                    del constant["ema_warmup_alpha"], constant["ema_warmup_tasks"]
                else:
                    constant.update(change)
                torch.save(data, Path(directory) / "max-va.pth")
                with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                    baseline.MyLearner().load(directory)

    def test_training_lifecycle_counts_tasks_not_optimizer_steps(self):
        config = copy.deepcopy(self.config)
        config["constant_condition"]["ema_warmup_tasks"] = 3
        config["experiment_config"] = dict(train_iterations=6, validation_tasks=1, validate_every=2)
        x = torch.randn(6, 3)
        y = torch.tensor([0, 1, 0, 1, 0, 1])
        tasks = [SimpleNamespace(num_ways=2, support_set=(x + i, y, y),
                                  query_set=(x - i - .2, y, y)) for i in range(6)]
        with patch.object(baseline, "ResNet", Tiny), patch.object(baseline, "read_config", return_value=config), patch.object(
                torch.cuda, "is_available", return_value=False):
            meta = baseline.MyMetaLearner(2, 2, SimpleNamespace(log=lambda *a, **kw: None))
            t = meta.transport
            completed, backprop = [], []
            complete, query = t.complete_training_episode, baseline.query_loss
            expected = t.m.clone()
            def query_call(*args, **kwargs):
                result = query(*args, **kwargs)
                if t.training:
                    result[1].register_hook(lambda g: backprop.append(True))
                return result
            def commit(e):
                self.assertEqual(len(backprop), len(completed) + 1)
                self.assertEqual(t.ema_completed_tasks.item(), len(completed))
                if not completed:
                    expected.copy_(e)
                elif len(completed) < 3:
                    expected.mul_(.9).add_(e, alpha=.1)
                else:
                    expected.mul_(.99).add_(e, alpha=1 - .99)
                complete(e)
                completed.append(e.clone())
                self.assertTrue(torch.equal(t.m, expected))
                self.assertEqual(t.ema_completed_tasks.item(), len(completed))
            with patch.object(t, "complete_training_episode", side_effect=commit), patch.object(
                    baseline, "query_loss", side_effect=query_call), patch.object(
                    meta.optimizer, "step", wraps=meta.optimizer.step) as optimizer_step:
                meta.meta_fit(lambda n: iter(tasks[:n]), lambda n: iter([tasks[0]] * n))
            self.assertEqual(len(completed), 6)
            self.assertEqual(optimizer_step.call_count, 3)
            self.assertEqual(t.ema_completed_tasks.item(), 6)

    def test_5k_logging_keeps_required_metrics_and_does_not_reset_schedule(self):
        self.assertEqual(self.config["experiment_config"]["validate_every"], 5000)
        t = self.transport()
        t.complete_training_episode(torch.ones_like(t.m))
        condition = t.condition()
        name, p = next(self.encoder.named_parameters())
        t.transport_gradient(name, torch.ones_like(p), condition)
        before = t.ema_completed_tasks.clone()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            log_metrics(SimpleNamespace(logs_dir=directory), t, 5000)
            record = json.loads((Path(directory) / "lrsg_metrics.jsonl").read_text())
        self.assertEqual(record["iteration"], 5000)
        self.assertTrue({"tc/low_rank_delta_abs_mean", "lrsg/u_norm",
                         "tc/low_rank_delta_task_std_mean", "lrsg/correction_to_gradient_ratio",
                         "lrsg/v_norm"}.issubset(record))
        self.assertTrue(torch.equal(before, t.ema_completed_tasks))


if __name__ == "__main__":
    unittest.main()
