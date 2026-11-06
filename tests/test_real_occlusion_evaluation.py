"""Test causal history and future timing in approximate repeat scoring."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from prediction.data import Normalization
from tools.evaluate_real_occlusion_repeats import evaluate


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.ones(()))

    def forward(self,x,rollout_steps):
        # The prediction depends only on the final input observation.
        return torch.stack([x[:,-1,:1]*self.weight + i for i in range(rollout_steps+1)],dim=1)


class RealOcclusionEvaluationTests(unittest.TestCase):
    def test_current_and_six_step_future_with_matching_support(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); prepared=root/'prepared'; prepared.mkdir(); train=root/'train'; train.mkdir()
            depth={'fx':390}; provenance={'depth_camera':depth,'depth_scale_m_per_unit':.0001}
            (train/'manifest.json').write_text(json.dumps({'provenance':provenance}))
            (prepared/'report.json').write_text(json.dumps({'training_depth_calibration':provenance,
                'test_camera':{'depth':depth,'depth_scale_m_per_unit':.0001}}))
            shape=(10,23,45); height=np.ones(shape,np.float32); valid=np.ones(shape,bool)
            hidden=np.zeros((23,45),bool); hidden[:,20:]=True; valid[:,:,20:]=False
            np.savez_compressed(prepared/'trial.npz',blocked_height_mm=height,blocked_valid=valid,
                blocked_timestamp_ms=np.arange(10)*1000/30,repeat_height_mm=height,repeat_valid=np.ones(shape,bool),
                conservative_hidden=hidden,reference_index=np.arange(10))
            grid={'x_min_mm':-80,'x_max_mm':100,'y_min_mm':-28,'y_max_mm':64,'cell_size_mm':4,
                  'z_min_mm':230,'z_max_mm':450,'min_points_per_cell':1,'aggregation':'mean'}
            checkpoint={'shape':(23,45),'geometry':{'frame_name':'camera_depth_frame_mm_not_printer_height',
                        'camera':{'width':640,'height':480,'fps':30},'grid':grid,'camera_to_output':np.eye(4).tolist()},
                        'train_sessions':[str(train)]}
            config={'history':2,'expected_dt_ms':1000/30,'timing_tolerance':.2,'age_scale_ms':1000}
            with patch('tools.evaluate_real_occlusion_repeats.load_checkpoint',return_value=(DummyModel(),config,Normalization(0,1),checkpoint)):
                evaluate(prepared,root/'dummy.pt',root/'out','cpu',1,4)
            result=json.loads((root/'out'/'metrics.json').read_text())['patterns']['trial']
            self.assertEqual(result['scored_windows'],9)
            self.assertEqual(result['future_scored_windows'],3)
            self.assertEqual(result['persistence_hidden_repeat_error']['pixels'],0)
            self.assertEqual(result['model_on_persistence_hidden_support']['pixels'],0)
            self.assertEqual(result['persistence_visible_repeat_error']['mae_mm'],0.)
            self.assertEqual(result['future_model_on_persistence_visible_support']['mae_mm'],6.)


if __name__=='__main__':
    unittest.main()
