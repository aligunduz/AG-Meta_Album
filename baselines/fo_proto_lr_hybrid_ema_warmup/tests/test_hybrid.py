import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import model as baseline
from task_transport import ConstantConditionedTransport
from helpers_fo_proto_constz_lrsgmaml import adapt, query_loss
from embedding_log import EpisodeWriter, record_episode, measures
from analyze_embeddings import analyze

class Tiny(nn.Module):
    in_features=512
    def __init__(self,**kwargs):
        super().__init__(); self.encoder=nn.Linear(3,512)
    def forward_weights(self,x,weights,embedding=False):
        z=F.linear(x,weights[0],weights[1]).tanh()/512**.5
        return z if embedding else F.linear(z,weights[-2],weights[-1])

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(17)
        self.cfg=baseline.read_config(); self.model=Tiny()
        self.t=ConstantConditionedTransport(self.model,self.cfg)
        self.x=torch.randn(6,3); self.y=torch.tensor([0,1,0,1,0,1]); self.q=torch.randn(6,3)
        self.task=SimpleNamespace(support_set=(self.x,self.y,self.y),query_set=(self.q,self.y,self.y),num_ways=2,num_shots=3,dataset="A",original_class_idx=np.array([3,7]))
    def run_task(self,t=None):
        t=self.t if t is None else t
        fast,z=adapt(self.model,list(self.model.parameters()),self.x,self.y,self.cfg["method_config"],2,t,return_task_embedding=True)
        out,loss=query_loss(self.model,fast,self.q,self.y)
        return out,loss,z
    def test_formula_endpoints(self):
        self.t.complete_training_episode(torch.ones(512))
        for alpha in (0.,.5,1.):
            self.t.alpha=alpha
            self.t.condition(torch.full((512,),3.))
            np.testing.assert_allclose(self.t.observation["vectors"]["z_hybrid"],1+2*alpha)
    def test_first_and_boundary(self):
        with patch.object(self.t.gate_net,"forward",wraps=self.t.gate_net.forward) as gate:
            out,loss,z=self.run_task(); self.assertEqual(gate.call_count,0)
        self.assertFalse(self.t.m_initialized); loss.backward()
        self.t.complete_training_episode(z); torch.testing.assert_close(self.t.m,z)
        self.t.ema_completed_tasks.fill_(4999); old=self.t.m.clone()
        self.t.condition(torch.ones(512)); torch.testing.assert_close(self.t.m,old)
        self.t.complete_training_episode(torch.ones(512)); torch.testing.assert_close(self.t.m,.9*old+.1)
        old=self.t.m.clone(); self.t.complete_training_episode(torch.zeros(512)); torch.testing.assert_close(self.t.m,.99*old)
    def test_eval_frozen_and_support_sensitive(self):
        self.t.complete_training_episode(torch.ones(512)); self.t.eval()
        old=copy.deepcopy(self.t.state_dict())
        for phase in ("validation","test"):
            self.run_task(); self.t.complete_training_episode(torch.zeros(512))
            for k,v in old.items(): torch.testing.assert_close(v,self.t.state_dict()[k])
        self.t.condition(torch.zeros(512)); a=self.t.observation["vectors"]["z_hybrid"].copy()
        self.t.condition(torch.ones(512)); self.assertFalse(np.array_equal(a,self.t.observation["vectors"]["z_hybrid"]))
    def test_logging_equivalence_and_gradients(self):
        self.t.complete_training_episode(torch.ones(512))
        with torch.no_grad():
            self.t.u["0"].normal_(0,.1); self.t.gate_net.out.weight.normal_(0,.1)
            self.t.gate_net.hidden.bias.fill_(1)
        before=torch.get_rng_state().clone()
        a,loss,z=self.run_task(); loss.backward()
        grads=[p.grad.clone() for p in list(self.model.parameters())+list(self.t.parameters())]
        for p in list(self.model.parameters())+list(self.t.parameters()):
            self.assertIsNotNone(p.grad); self.assertTrue(torch.isfinite(p.grad).all()); p.grad=None
        self.t.capture=False; b,loss,_=self.run_task(); loss.backward()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for g,p in zip(grads,list(self.model.parameters())+list(self.t.parameters())): torch.testing.assert_close(g,p.grad,rtol=0,atol=0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for group in (self.model.parameters(),self.t.u.values(),self.t.v.values(),self.t.gate_net.parameters()):
            self.assertTrue(any(p.grad.abs().sum()>0 for p in group))
    def test_chunks_metadata_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            self.cfg["embedding_logging"]["chunk_size"]=2
            writer=EpisodeWriter(d,self.cfg)
            state=torch.get_rng_state().clone()
            for i in range(3):
                out,loss,z=self.run_task(); record_episode(writer,self.t,self.task,"train",i+1,None,out,loss); self.t.complete_training_episode(z)
            writer.flush(); self.assertTrue(torch.equal(state,torch.get_rng_state()))
            rows=[]
            for f in sorted(Path(d).glob("*.jsonl")):
                with np.load(f.with_suffix(".npz")) as arrays:
                    for line in f.read_text().splitlines():
                        r=json.loads(line); rows.append(r); self.assertEqual(arrays["record_id"][r["vector_row"]],r["record_id"])
                        self.assertEqual(arrays["z_task"].dtype,np.float32); self.assertIsNone(r["domain_name"])
            self.assertEqual(len(rows),3); self.assertEqual(rows[0]["invalid_metric_count"],3)
            new=EpisodeWriter(d,self.cfg); self.assertEqual(new.next_id,3)
            report=analyze(d,Path(d)/"analysis.json",max_tasks=3,max_pairs=5)
            self.assertTrue(report["groups"]); self.assertTrue(report["ema_movement"])
    def test_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as d, patch.object(baseline,"make_encoder",side_effect=lambda args:Tiny()):
            logger=SimpleNamespace(logs_dir=str(Path(d)/"logs"),log=lambda *args,**kwargs:None)
            a=baseline.MyMetaLearner(2,2,logger)
            a.transport.complete_training_episode(torch.ones(512)); a.training_step=1
            a.resume_buffer=[torch.ones_like(p) for p in a.meta_parameters]
            a.save_training_checkpoint(Path(d)/"resume.pth",{"next_episode":1})
            b=baseline.MyMetaLearner(2,2,logger)
            self.assertEqual(b.load_training_checkpoint(Path(d)/"resume.pth"),{"next_episode":1})
            for t in (a.transport,b.transport): t.complete_training_episode(torch.full((512,),3.))
            for k,v in a.transport.state_dict().items(): torch.testing.assert_close(v,b.transport.state_dict()[k])
            for x,y in zip(a.resume_buffer,b.resume_buffer): torch.testing.assert_close(x,y)
            state=baseline.snapshot(a.meta_learner,a.transport); state["training_step"]=1
            learner=baseline.MyLearner(a.model_args,state,a.config,.5); learner.save(d)
            loaded=baseline.MyLearner(); loaded.load(d)
            self.assertEqual(loaded.transport.alpha,.5); self.assertEqual(loaded.transport.ema_completed_tasks.item(),2)
    def test_training_validation_test_integration(self):
        with tempfile.TemporaryDirectory() as d, patch.object(baseline,"make_encoder",side_effect=lambda args:Tiny()), patch.object(baseline,"read_config",return_value=copy.deepcopy(self.cfg)):
            owner=baseline.MyMetaLearner(2,2,SimpleNamespace(logs_dir=str(Path(d)/"logs"),log=lambda *a,**k:None))
            owner.set_run_context(data_seed=95)
            owner.config["experiment_config"].update(train_iterations=2,validate_every=2,validation_tasks=1)
            learner=owner.meta_fit(lambda count:iter([self.task]*count),lambda count:iter([self.task]*count))
            self.assertEqual(owner.transport.ema_completed_tasks.item(),2)
            model_dir=Path(d)/"model"; model_dir.mkdir(exist_ok=True); learner.save(model_dir)
            loaded=baseline.MyLearner(); loaded.load(model_dir)
            predictor=loaded.fit((*self.task.support_set,2,3))
            before=loaded.transport.m.clone()
            probabilities=predictor.predict(self.q)
            loaded.record_test_episode(self.task,probabilities,1)
            torch.testing.assert_close(before,loaded.transport.m)
            rows=[json.loads(line) for f in (Path(d)/"model"/"embeddings").glob("*.jsonl") for line in f.read_text().splitlines()]
            self.assertEqual(sorted(r["phase"] for r in rows),["test","train","train","validation"])
            self.assertEqual(len({r["record_id"] for r in rows}),4)
            self.assertTrue(all(r["data_seed"] == 95 for r in rows))
            # The unchanged Colab notebook copies/artifacts the entire model directory.
            import shutil
            shutil.copytree(model_dir, Path(d)/"drive"/"model")
            copied = Path(d)/"drive"/"model"/"embeddings"
            self.assertEqual(len(list(copied.glob("*.npz"))),len(list((model_dir/"embeddings").glob("*.npz"))))
            self.assertEqual(json.loads((copied/"run.json").read_text())["data_seed"],95)
            self.assertEqual([r["training_step"] for r in rows if r["phase"]=="test"],[2])

    def test_resume_matches_next_optimizer_update(self):
        with tempfile.TemporaryDirectory() as d, patch.object(baseline,"make_encoder",side_effect=lambda args:Tiny()):
            logger=SimpleNamespace(logs_dir=str(Path(d)/"logs"),log=lambda *a,**k:None)
            a=baseline.MyMetaLearner(2,2,logger)
            def step(owner):
                fast,z=adapt(owner.meta_learner,owner.weights,self.x,self.y,owner.params,2,owner.transport,return_task_embedding=True)
                _,loss=query_loss(owner.meta_learner,fast,self.q,self.y)
                loss.backward(); owner.optimizer.step(); owner.optimizer.zero_grad(set_to_none=True)
                owner.transport.complete_training_episode(z)
            step(a); a.save_training_checkpoint(Path(d)/"resume.pth")
            b=baseline.MyMetaLearner(2,2,logger); b.load_training_checkpoint(Path(d)/"resume.pth")
            step(a); step(b)
            for x,y in zip(a.meta_parameters,b.meta_parameters): torch.testing.assert_close(x,y,rtol=0,atol=0)
            torch.testing.assert_close(a.transport.m,b.transport.m,rtol=0,atol=0)

    def test_invalid_alpha(self):
        for alpha in (-1,2,True,float("nan")):
            self.cfg["hybrid"]["alpha"]=alpha
            with self.assertRaises(ValueError): ConstantConditionedTransport(self.model,self.cfg)

if __name__=="__main__": unittest.main()
