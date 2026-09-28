"""Frozen shared encoder/LRSG; independently trained GateNet/EMA branches."""
from copy import deepcopy
from itertools import zip_longest
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from .source import reference, ROOT
from .data import DOMAIN_INDEX, TAXONOMY, file_hash, json_hash, episode_record, context_for_task
from .reporting import atomic_write, write_summary, error_record


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def options_from(config, options_path, data_seed):
    options = dict(config)
    if options_path:
        supplied = json.loads(Path(options_path).read_text(encoding="utf-8"))
        unknown = set(supplied) - set(options)
        if unknown:
            raise ValueError(f"Unknown stage2 options: {sorted(unknown)}")
        options.update(supplied)
    if type(options["data_seed"]) is not int or options["data_seed"] != data_seed:
        raise ValueError("--data_seed and options data_seed must agree (stage-1 split seed)")
    for key, default in (("stage2_seed", 100000 + data_seed),
                         ("validation_seed", data_seed), ("test_seed", data_seed)):
        if options[key] is None:
            options[key] = default
        if type(options[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    for key in ("budget_per_domain", "warmup_tasks", "bootstrap_samples"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("warmup_alpha", "ema_alpha"):
        if not 0 < options[key] <= 1:
            raise ValueError(f"Invalid {key}")
    for key in ("atol", "rtol"):
        if not math.isfinite(options[key]) or options[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not options.get("source_checkpoint"):
        raise ValueError("Set source_checkpoint to the stage-1 EMA warmup max-va.pth")
    return options


def source_checkpoint(options):
    path = Path(options["source_checkpoint"]).resolve()
    if path.name != "max-va.pth":
        raise ValueError("Stage 1 must be a selected max-va.pth checkpoint")
    data = torch.load(path, map_location="cpu", weights_only=True)
    reference().validate_config(data["config"])
    constant = data["config"]["constant_condition"]
    if constant["init"] != "ema" or "ema_warmup_tasks" not in constant:
        raise ValueError("Source must be the EMA warmup baseline, not learned-LR/zero/shuffle")
    if data["method"] != "fo-proto-constz-lrsgmaml" or data["format_version"] != 1:
        raise ValueError("Wrong source checkpoint format")
    if data["config"]["validation_datasets"] != 3:
        raise ValueError("Source must use the FEEDBACK 7/3 split")
    if not bool(data["state"]["lrsg"]["m_initialized"]):
        raise ValueError("Source EMA is uninitialized")
    return data


class Branch:
    """Only GateNet is optimized. Shared map is always the frozen reference."""
    def __init__(self, shared, trainable=False):
        self.shared = shared
        self.names = shared.names
        self.gate_net = deepcopy(shared.gate_net)
        self.gate_net.requires_grad_(trainable)
        self.gate_net.eval()
        self.m = shared.m.detach().clone()
        self.initialized = True
        self.tasks = self.steps = self.pending = 0
        self.optimizer = None
        self.buffer = None

    def condition(self):
        values = self.gate_net(self.m.detach().clone())
        return (values[:len(self.names)] * self.shared.scalar_scale,
                {row["key"]: values[row["start"]:row["stop"]] * self.shared.low_rank_scale
                 for row in self.shared.rank_layout})

    def transport_gradient(self, name, grad, conditioning):
        return self.shared.transport_gradient(name, grad, conditioning)

    @torch.no_grad()
    def delta(self):
        _, coefficients = self.condition()
        return torch.cat([coefficients[row["key"]] for row in self.shared.rank_layout])

    def configure_optimizer(self, method):
        self.optimizer = torch.optim.Adam(self.gate_net.parameters(), lr=method["outer_lr"])
        self.buffer = [torch.zeros_like(p) for p in self.gate_net.parameters()]

    def step(self):
        if not self.pending:
            return
        for parameter, grad in zip(self.gate_net.parameters(), self.buffer):
            parameter.grad = grad
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.buffer = [torch.zeros_like(p) for p in self.gate_net.parameters()]
        self.pending = 0
        self.steps += 1

    @torch.no_grad()
    def complete(self, embedding, options):
        self.tasks += 1
        alpha = options["warmup_alpha"] if self.tasks <= options["warmup_tasks"] else options["ema_alpha"]
        self.m.mul_(1 - alpha).add_(embedding.detach(), alpha=alpha)

    def state(self):
        return dict(gate_net=cpu_state(self.gate_net), m=self.m.detach().cpu().clone(),
                    initialized=self.initialized, tasks=self.tasks, steps=self.steps,
                    pending=self.pending)

    def load(self, state):
        self.gate_net.load_state_dict(state["gate_net"], strict=True)
        self.m.copy_(state["m"].to(self.m))
        self.initialized = state["initialized"]
        self.tasks, self.steps, self.pending = state["tasks"], state["steps"], state["pending"]
        if not self.initialized:
            raise ValueError("Stage2 branch must inherit an initialized m0")


class Experiment:
    def __init__(self, source, options, manifest, provenance=None):
        self.source = source
        self.options, self.manifest = options, manifest
        self.ref = reference()
        self.base = self.ref.MyLearner(source["model_args"], source["state"],
                                       source["config"], source["best_validation_accuracy"])
        self.encoder, self.shared = self.base.learner, self.base.transport
        self.encoder.eval()
        # Keep encoder leaves differentiable for the functional inner loop.
        self.shared.requires_grad_(False)
        self.shared.eval()
        self.method = source["config"]["method_config"]
        self.seen = [domain for domain in DOMAIN_INDEX
                     if domain in {row["domain"] for row in manifest["train"]}]
        if len(self.seen) != 7:
            raise ValueError("Expected seven training domains")
        self.branches = {"G0": Branch(self.shared)}
        for domain in self.seen:
            self.branches[self.specialist_key(domain)] = Branch(self.shared, True)
        self.branches["all"] = Branch(self.shared, True)
        self.early = None
        self.provenance = provenance or self.make_provenance()
        self.validation = {}
        self.train_summary = {}
        initial = self.branches["G0"].delta()
        for branch in self.branches.values():
            torch.testing.assert_close(branch.delta(), initial, rtol=0, atol=0)

    @staticmethod
    def specialist_key(domain):
        return f"specialist_{DOMAIN_INDEX[domain]}"

    def make_provenance(self):
        stored = dict(self.source.get("provenance", {}))
        for key in ("data_seed", "split_manifest", "validation_seed", "image_size",
                    "validation_manifest_sha256", "accuracy_aggregation"):
            if key in self.source:
                if key in stored and stored[key] != self.source[key]:
                    raise ValueError(f"Conflicting embedded provenance: {key}")
                stored[key] = self.source[key]
        if self.options.get("source_provenance"):
            extra = json.loads(Path(self.options["source_provenance"]).read_text())
            for key in set(extra) & set(stored):
                if extra[key] != stored[key]:
                    raise ValueError(f"Conflicting provenance: {key}")
            stored.update(extra)
        expected = dict(data_seed=self.options["data_seed"], split_manifest=self.manifest,
                        validation_seed=self.options["validation_seed"],
                        source_sha256=file_hash(self.options["source_checkpoint"]))
        for key in expected.keys() & stored.keys():
            if expected[key] != stored[key]:
                raise ValueError(f"Source provenance mismatch: {key}")
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        repository = ROOT.parents[1]
        files = list(ROOT.glob("*.py")) + list(Path(__file__).parent.glob("*.py"))
        files += [repository / name for name in (
            "cdmetadl/ingestion/data_generator.py", "cdmetadl/ingestion/image_dataset.py",
            "cdmetadl/ingestion/ingestion.py", "cdmetadl/helpers/general_helpers.py")]
        code_hashes = {p.relative_to(repository).as_posix(): file_hash(p) for p in files}
        if "code_sha256" in stored:
            for name, expected_hash in stored["code_sha256"].items():
                if name not in code_hashes or code_hashes[name] != expected_hash:
                    raise ValueError(f"Source code provenance mismatch: {name}")
        return dict(source_checkpoint=str(Path(self.options["source_checkpoint"]).resolve()),
                    source_sha256=expected["source_sha256"], code_revision=revision,
                    source_code_revision=stored.get("code_revision"),
                    source_code_hashes_verified="code_sha256" in stored,
                    code_sha256=code_hashes,
                    data_seed=self.options["data_seed"],
                    data_seed_verified="data_seed" in stored,
                    source_provenance=stored, split_manifest=self.manifest,
                    seeds={key: self.options[key] for key in
                           ("data_seed", "stage2_seed", "validation_seed", "test_seed")})

    def set_data_provenance(self, root, image_size):
        stored = self.provenance["source_provenance"]
        if "image_size" in stored and stored["image_size"] != image_size:
            raise ValueError("Source image_size provenance mismatch")
        self.provenance.update(image_size=image_size, input_data_dir=str(Path(root).resolve()),
                               preprocessing="ImageDataset: Resize((size,size)), ToTensor; source functional BN",
                               source_config_sha256=json_hash(self.source["config"]),
                               torch_version=str(torch.__version__))

    def route(self, domain, condition="B"):
        if domain not in DOMAIN_INDEX:
            raise ValueError(f"Unknown domain {domain!r}")
        if condition == "A" or domain not in self.seen:
            return "G0"
        if condition == "B":
            return self.specialist_key(domain)
        if condition == "C":
            return "all"
        if condition == "D":
            if self.early is None:
                raise ValueError("Missing stepmatched snapshot")
            return "all_stepmatched"
        raise ValueError(f"Unknown condition: {condition}")

    def branch(self, key):
        return self.early if key == "all_stepmatched" else self.branches[key]

    def adapt(self, support_set, key, training=False):
        support, labels, _, ways, _ = support_set
        return self.ref.adapt(self.encoder, list(self.encoder.parameters()),
                              support.to(self.base.dev), labels.to(self.base.dev),
                              self.method, ways, self.branch(key), return_task_embedding=training)

    def evaluate(self, task, key):
        fast = self.adapt((*task.support_set, task.num_ways, task.num_shots), key)
        with torch.no_grad():
            logits, loss = self.ref.query_loss(self.encoder, fast, task.query_set[0].to(self.base.dev),
                                               task.query_set[1].to(self.base.dev))
        return logits.detach().cpu(), loss.item()

    def train_task(self, task, key):
        branch = self.branch(key)
        fast, embedding = self.adapt((*task.support_set, task.num_ways, task.num_shots), key, True)
        logits, loss = self.ref.query_loss(self.encoder, fast, task.query_set[0].to(self.base.dev),
                                          task.query_set[1].to(self.base.dev))
        grads = torch.autograd.grad(loss, tuple(branch.gate_net.parameters()))
        for buffer, grad in zip(branch.buffer, grads):
            if self.method["grad_clip"] is not None:
                grad = grad.clamp(-self.method["grad_clip"], self.method["grad_clip"])
            buffer.add_(grad.detach())
        branch.pending += 1
        if branch.pending == self.method["meta_batch_size"]:
            branch.step()
        branch.complete(embedding, self.options)
        return dict(loss=loss.item(), correct=int((logits.argmax(1).cpu() == task.query_set[1]).sum()),
                    total=task.query_set[1].numel())

    def assert_fixed(self):
        for module, state in ((self.encoder, self.source["state"]["encoder"]),
                              (self.shared, self.source["state"]["lrsg"])):
            for key, value in module.state_dict().items():
                if not torch.equal(value.cpu(), state[key].cpu()):
                    raise AssertionError(f"Frozen parameter/buffer changed: {key}")
        for parameter in self.encoder.parameters():
            if parameter.grad is not None:
                raise AssertionError("Encoder outer .grad accumulated")
        fixed = self.branches["G0"]
        if not torch.equal(fixed.m, self.shared.m) or fixed.tasks != 0 or fixed.steps != 0:
            raise AssertionError("G0/m0 changed")
        for a, b in zip(fixed.gate_net.parameters(), self.shared.gate_net.parameters()):
            if not torch.equal(a, b):
                raise AssertionError("G0 parameters changed")

    def diagnostics(self):
        ref_delta = self.branches["G0"].delta()
        values = {}
        branches = dict(self.branches)
        if self.early is not None:
            branches["all_stepmatched"] = self.early
        for name, branch in branches.items():
            delta = branch.delta()
            n = branch.tasks
            values[name] = dict(tasks=n, adam_steps=branch.steps, pending=branch.pending,
                                initialized=branch.initialized,
                                ema_alpha=0. if n == 0 else (self.options["warmup_alpha"]
                                  if n <= self.options["warmup_tasks"] else self.options["ema_alpha"]),
                                delta_count=delta.numel(), delta_norm=delta.norm().item(),
                                delta_abs_mean=delta.abs().mean().item(),
                                delta_distance=(delta - ref_delta).norm().item())
        return values

    def save(self, path):
        # Preserve trained weights before running any final verification.
        # The hash-linked training summary records the final pass/fail result.
        path = Path(path)
        early_path = path.with_name("stage2-all-stepmatched.pth")
        if path.exists() or early_path.exists():
            raise FileExistsError("Stage2 checkpoints already exist; choose a new output directory")
        data = dict(format_version=1, method="fo-proto-domain-ema-stage2", source=self.source,
                    verification_status="pending",
                    recovery_shared_state=dict(encoder=cpu_state(self.encoder), lrsg=cpu_state(self.shared)),
                    options=self.options, manifest=self.manifest, taxonomy=TAXONOMY,
                    domain_to_index=DOMAIN_INDEX,
                    branches={key: branch.state() for key, branch in self.branches.items()},
                    all_stepmatched=self.early.state(), provenance=self.provenance,
                    validation=deepcopy(self.validation), training=deepcopy(self.train_summary))
        atomic_write(path, lambda handle: torch.save(data, handle))
        early = dict(branch=self.early.state(), source_sha256=self.provenance["source_sha256"],
                     rank_layout=self.shared.rank_layout, taxonomy=TAXONOMY)
        atomic_write(early_path, lambda handle: torch.save(early, handle))

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (data["method"] != "fo-proto-domain-ema-stage2" or data["format_version"] != 1
                or data["taxonomy"] != TAXONOMY or data["domain_to_index"] != DOMAIN_INDEX):
            raise ValueError("Stage2 checkpoint method/taxonomy mismatch")
        if data.get("verification_status") == "pending":
            report_path = path.with_name("summary.json")
            if not report_path.is_file():
                raise ValueError("Checkpoint is preserved but verification is pending; keep its training summary.json beside it")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (report.get("kind") != "stage2_training" or report.get("status") != "passed"
                    or report.get("checkpoint", {}).get("sha256") != file_hash(path)):
                raise ValueError("Checkpoint verification failed, is incomplete, or its summary hash does not match")
            data["validation"], data["training"] = report["validation"], report["training"]
        obj = cls(data["source"], data["options"], data["manifest"], data["provenance"])
        if set(data["branches"]) != set(obj.branches):
            raise ValueError("Checkpoint specialist set mismatch")
        for name, state in data["branches"].items():
            obj.branches[name].load(state)
            obj.branches[name].gate_net.requires_grad_(False)
        obj.early = Branch(obj.shared)
        obj.early.load(data["all_stepmatched"])
        obj.validation, obj.train_summary = data["validation"], data["training"]
        obj.assert_fixed()
        return obj


def compare_logits(actual, expected, options, task_id):
    torch.testing.assert_close(actual, expected, atol=options["atol"], rtol=options["rtol"],
                               msg=f"Logits mismatch on {task_id}")
    if not torch.equal(actual.argmax(1), expected.argmax(1)):
        raise AssertionError(f"Prediction mismatch on {task_id}")
    return float((actual - expected).abs().max())


def source_logits(learner, task):
    predictor = learner.fit((*task.support_set, task.num_ways, task.num_shots))
    with torch.no_grad():
        return predictor.model.forward_weights(task.query_set[0].to(predictor.dev),
                                                predictor.weights).cpu()


def validation_check(experiment, generator, output, label, reference_learner):
    start = time.perf_counter()
    correct = total = 0
    records, max_error = [], 0.
    count = experiment.source["config"]["experiment_config"]["validation_tasks"]
    with Path(output).open("x", encoding="utf-8") as handle:
        for task in generator(count):
            if not task.reference_sampler_verified:
                raise AssertionError("Validation task has not been checked against the reference sampler")
            record = episode_record(task)
            actual, loss = experiment.evaluate(task, "G0")
            expected = source_logits(reference_learner, task)
            max_error = max(max_error, compare_logits(actual, expected, experiment.options, record["task_id"]))
            pred = actual.argmax(1)
            correct += int((pred == task.query_set[1]).sum())
            total += pred.numel()
            record.update(predictions=pred.tolist(), loss=loss, logits=actual.tolist(),
                          logits_sha256=json_hash(actual.tolist()))
            records.append(record)
            handle.write(json.dumps(record) + "\n")
    if not total:
        raise ValueError("No validation examples")
    manifest_hash = json_hash([r["task_id"] for r in records])
    stored = experiment.provenance["source_provenance"]
    historical = (experiment.provenance["data_seed_verified"] and
                  stored.get("validation_manifest_sha256") == manifest_hash and
                  stored.get("accuracy_aggregation") == "query_micro")
    if ("validation_manifest_sha256" in stored and
            stored["validation_manifest_sha256"] != manifest_hash):
        raise ValueError("Supplied historical validation manifest does not match reproduced tasks")
    if historical and not math.isclose(correct / total, experiment.source["best_validation_accuracy"],
                                       rel_tol=experiment.options["rtol"], abs_tol=experiment.options["atol"]):
        raise AssertionError("Verified historical validation score differs")
    result = dict(accuracy=correct / total, correct=correct, total=total, tasks=len(records),
                  manifest_sha256=manifest_hash, result_sha256=json_hash(records),
                  max_logit_error=max_error, reference_match=True,
                  reference_sampler_match=True,
                  historical_score_verified=historical,
                  historical_note="verified" if historical else "Historical episode identity/provenance unavailable; source/new same-manifest comparison only",
                  seconds=time.perf_counter() - start, adaptation_operations=2 * len(records))
    experiment.validation[label] = result
    experiment.assert_fixed()
    return result


def compare_validation_outputs(before_path, after_path, options):
    """Exact episode/prediction identity; tolerance-based numeric comparison."""
    count, max_logit_error, max_loss_error = 0, 0., 0.
    with Path(before_path).open(encoding="utf-8") as before, Path(after_path).open(encoding="utf-8") as after:
        for left, right in zip_longest(before, after):
            if left is None or right is None:
                raise AssertionError("Before/after validation task counts differ")
            old, new = json.loads(left), json.loads(right)
            identity = old["task_id"]
            if identity != new["task_id"]:
                raise AssertionError(f"Before/after validation manifest differs at task {count + 1}")
            if old["predictions"] != new["predictions"]:
                raise AssertionError(f"Before/after validation predictions differ: {identity}")
            error = compare_logits(torch.tensor(new["logits"], dtype=torch.float64),
                                   torch.tensor(old["logits"], dtype=torch.float64), options, identity)
            max_logit_error = max(max_logit_error, error)
            if (not math.isfinite(new["loss"]) or not math.isfinite(old["loss"]) or
                    not math.isclose(new["loss"], old["loss"], abs_tol=options["atol"], rel_tol=options["rtol"])):
                raise AssertionError(f"Before/after validation loss differs: {identity}")
            max_loss_error = max(max_loss_error, abs(new["loss"] - old["loss"]))
            count += 1
    return dict(status="passed", tasks=count, max_logit_error=max_logit_error,
                max_loss_error=max_loss_error, atol=options["atol"], rtol=options["rtol"],
                manifest_match=True, predictions_match=True)


def train(experiment, generator, valid_generator, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Training summary already exists; use a new run directory")
    report = dict(kind="stage2_training", status="running", phase="initial_validation",
                  provenance=experiment.provenance, options=experiment.options)
    write_summary(output / "summary.json", report)
    try:
        _train(experiment, generator, valid_generator, output, report)
        report.update(status="passed", phase="complete")
    except BaseException as error:
        report.update(status="failed", error=error_record(error))
        raise
    finally:
        report.update(training=experiment.train_summary, validation=experiment.validation)
        # Save the outcome before propagating verification errors. No expensive
        # numeric checks are performed in this persistence path.
        write_summary(output / "summary.json", report)


def _train(experiment, generator, valid_generator, output, report):
    # The actual baseline loader is the comparison oracle, including config.
    oracle = reference().MyLearner()
    oracle.load(experiment.options["source_checkpoint"])
    before = validation_check(experiment, valid_generator, output / "validation-before.jsonl", "before", oracle)
    report["phase"] = "training"
    start = time.perf_counter()
    trainable = {k: b for k, b in experiment.branches.items() if k != "G0"}
    for branch in trainable.values():
        branch.configure_optimizer(experiment.method)
    budget = experiment.options["budget_per_domain"]
    target = len(experiment.seen) * budget
    accepted = skipped = 0
    verified_initial = set()
    initial_checks = []
    diagnostic_operations = 0
    diagnostic_seconds = 0.
    interval = experiment.source["config"]["experiment_config"]["validate_every"]
    with (output / "training-tasks.jsonl").open("x", encoding="utf-8") as tasks_log, \
            (output / "training-metrics.jsonl").open("x", encoding="utf-8") as metrics_log:
        # One infinite generator invocation: RNG is never reset at chunk boundaries.
        for task in generator(None):
            domain = context_for_task(task)["domain"]
            if domain not in experiment.seen:
                raise ValueError(f"Non-training domain in stage2 stream: {domain}")
            key = experiment.specialist_key(domain)
            if experiment.branches[key].tasks == budget:
                skipped += 1
                continue
            record = episode_record(task)
            if key not in verified_initial:
                check_start = time.perf_counter()
                initial, _ = experiment.evaluate(task, key)
                expected = source_logits(oracle, task)
                diagnostic_operations += 2
                error = compare_logits(initial, expected, experiment.options, record["task_id"])
                initial_checks.append(dict(branch=key, task_id=record["task_id"], max_logit_error=error,
                                           predictions_match=True,
                                           accuracy=float((initial.argmax(1) == task.query_set[1]).float().mean())))
                if not verified_initial:
                    control, _ = experiment.evaluate(task, "all")
                    diagnostic_operations += 1
                    error = compare_logits(control, expected, experiment.options, record["task_id"])
                    initial_checks.append(dict(branch="all", task_id=record["task_id"], max_logit_error=error,
                                               predictions_match=True,
                                               accuracy=float((control.argmax(1) == task.query_set[1]).float().mean())))
                verified_initial.add(key)
                diagnostic_seconds += time.perf_counter() - check_start
            record["specialist"] = dict(branch=key, **experiment.train_task(task, key))
            record["control"] = dict(branch="all", **experiment.train_task(task, "all"))
            accepted += 1
            # Explicit final partial-batch policy: flush unscaled SUM at budget.
            if experiment.branches[key].tasks == budget:
                experiment.branches[key].step()
            if accepted == budget:
                # Snapshot never changes training accumulation. For odd B its
                # pending task is recorded; no extra optimizer step is inserted.
                experiment.early = Branch(experiment.shared)
                experiment.early.load(experiment.branches["all"].state())
            tasks_log.write(json.dumps(record) + "\n")
            if accepted == target:
                experiment.branches["all"].step()
                break
            if accepted % interval == 0:
                experiment.assert_fixed()
                metrics = dict(accepted=accepted, skipped=skipped, branches=experiment.diagnostics())
                metrics_log.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
    experiment.train_summary = dict(accepted_tasks=accepted, skipped_tasks=skipped,
                                   episode_branch_operations=2 * accepted,
                                   training_seconds=time.perf_counter() - start - diagnostic_seconds,
                                   zero_step_check_seconds=diagnostic_seconds,
                                   zero_step_check_adaptation_operations=diagnostic_operations,
                                   zero_step_verified_specialists=sorted(verified_initial),
                                   zero_step_checks=initial_checks,
                                   partial_batch_policy="flush unscaled sum only at final branch budget",
                                   early_snapshot_policy="after B tasks; do not flush pending gradients",
                                   seen_domains=experiment.seen,
                                   training_manifest_sha256=file_hash(output / "training-tasks.jsonl"))
    report.update(status="validation_pending", phase="checkpoint_save")
    checkpoint = output / "stage2-final.pth"
    experiment.save(checkpoint)
    report.update(checkpoint=dict(path=checkpoint.name, sha256=file_hash(checkpoint)),
                  training=experiment.train_summary, validation=experiment.validation,
                  phase="post_training_checks")
    write_summary(output / "summary.json", report)
    batch = experiment.method["meta_batch_size"]
    for key, branch in trainable.items():
        tasks = target if key == "all" else budget
        if branch.tasks != tasks or branch.steps != math.ceil(tasks / batch) or branch.pending:
            raise AssertionError(f"Budget/optimizer count mismatch: {key}")
        for parameter in branch.gate_net.parameters():
            if int(branch.optimizer.state[parameter]["step"]) != branch.steps:
                raise AssertionError(f"Adam internal step count mismatch: {key}")
    experiment.assert_fixed()
    report["diagnostics"] = experiment.diagnostics()
    with (output / "training-metrics.jsonl").open("a", encoding="utf-8") as metrics_log:
        metrics_log.write(json.dumps(dict(accepted=accepted, skipped=skipped, branches=report["diagnostics"])) + "\n")
    after = validation_check(experiment, valid_generator, output / "validation-after.jsonl", "after", oracle)
    if before["manifest_sha256"] != after["manifest_sha256"]:
        raise AssertionError("Frozen validation manifest changed during stage2")
    experiment.validation["before_after"] = compare_validation_outputs(
        output / "validation-before.jsonl", output / "validation-after.jsonl", experiment.options)
    experiment.assert_fixed()
