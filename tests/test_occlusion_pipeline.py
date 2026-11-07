"""Integration checks for training artifacts and occlusion metrics."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional prediction dependencies are not installed')
class OcclusionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        from prediction.data import Normalization
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.splits = {name: [SimpleNamespace(path=self.root / name, geometry={'frame_name': 'test'}, shape=(3, 5))]
                       for name in ('train', 'validation', 'test')}
        self.normalization = Normalization(10., 2.)
        self.config = {'history': 2, 'horizon': 1, 'base_channels': 2, 'epochs': 1,
                       'batch_size': 1, 'patience': 1, 'expected_dt_ms': 10}
        inputs = np.zeros((2, 4, 3, 5), np.float32)
        inputs[:, 0] = 1
        inputs[:, 1] = 1
        inputs[:, 1, 0, 0] = 0  # Never observed; future can nevertheless have a label.
        inputs[:, 0, 0, 0] = 0
        inputs[-1, 1, 0, 1] = 0  # Hidden at origin, but seen previously.
        inputs[-1, 0, 0, 1] = 0
        targets = np.ones((2, 1, 3, 5), np.float32) * 2
        valid = np.ones_like(targets, bool)
        valid[0, 0, 0, 0] = False
        hidden_origin = np.zeros((1, 3, 5), bool)
        hidden_origin[0, 0, 1] = True
        self.sample = {'inputs': inputs, 'targets': targets, 'target_valid': valid,
                       'hidden_origin': hidden_origin}

    def dataset(self, sessions, config, normalization, seed, **kwargs):
        sample = self.sample

        class Dataset:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                if index != 0:
                    raise IndexError(index)
                return sample

            def summary(self):
                return {'windows': 1, 'seed': seed}

        return Dataset()

    def test_train_checkpoint_and_evaluate_hidden_and_baseline_support(self):
        from prediction import occlusion
        with patch.object(occlusion, 'load_splits', return_value=self.splits), \
             patch.object(occlusion, 'fit_normalization', return_value=self.normalization) as fit, \
             patch.object(occlusion, 'OcclusionDataset', side_effect=self.dataset):
            run = occlusion.train(self.root / 'splits.json', self.config, self.root / 'run', 'cpu')
            fit.assert_called_once_with(self.splits['train'])
            self.assertFalse(run['test_evaluated'])
            model, config, norm, checkpoint = occlusion.load_checkpoint(self.root / 'run/best.pt', 'cpu')
            self.assertEqual(config['horizon'], 1)
            self.assertEqual(norm, self.normalization)
            self.assertEqual(checkpoint['train_sessions'], [str((self.root / 'train').resolve())])
            report = occlusion.evaluate(self.root / 'run/best.pt', self.root / 'splits.json',
                                        self.root / 'evaluation', 'test', 'cpu')
            current = report['leads'][0]['regions']['hidden']
            future = report['leads'][1]['regions']['hidden']
            self.assertEqual(current['target_valid'], 1)
            self.assertEqual(current['target_coverage'], .5)
            self.assertEqual(future['target_valid'], 2)
            self.assertEqual(future['never_seen'], 1)
            self.assertEqual(future['baseline_supported'], 1)
            self.assertEqual(future['baseline_coverage'], .5)
            self.assertEqual(future['baseline']['mae_mm'], 2.)
            self.assertEqual(future['model_on_baseline_support']['valid_pixels'], 1)
            self.assertEqual(report['leads'][1]['nominal_lead_ms'], 10)
            with self.assertRaises(FileExistsError):
                occlusion.train(self.root / 'splits.json', self.config, self.root / 'run', 'cpu')
            self.splits['test'] = self.splits['train']
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                occlusion.evaluate(self.root / 'run/best.pt', self.root / 'splits.json',
                                   self.root / 'rejected', 'test', 'cpu')

    def test_initialize_from_copies_weights_and_records_source(self):
        import hashlib
        from prediction import occlusion
        with patch.object(occlusion, 'load_splits', return_value=self.splits), \
             patch.object(occlusion, 'fit_normalization', return_value=self.normalization), \
             patch.object(occlusion, 'OcclusionDataset', side_effect=self.dataset):
            occlusion.train(self.root / 'splits.json', self.config, self.root / 'source', 'cpu')
            source_path = self.root / 'source/best.pt'
            source_model, _, _, _ = occlusion.load_checkpoint(source_path, 'cpu')
            target_config = occlusion.read_config({**self.config, 'history': 3, 'horizon': 2,
                                                   'expected_dt_ms': 33.3, 'learning_rate': 3e-4})
            model = occlusion._model(target_config)
            record = occlusion._initialize_weights(model, target_config, source_path)
            for name, value in source_model.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value)
            self.assertEqual(record['source_sha256'], hashlib.sha256(source_path.read_bytes()).hexdigest())
            self.assertFalse(record['optimizer_state_transferred'])
            run = occlusion.train(self.root / 'splits.json', {**self.config, 'learning_rate': 3e-4},
                                  self.root / 'finetuned', 'cpu', initialize_from=source_path)
            self.assertEqual(run['initialization']['source_checkpoint'], str(source_path.resolve()))
            _, _, _, checkpoint = occlusion.load_checkpoint(self.root / 'finetuned/best.pt', 'cpu')
            self.assertEqual(checkpoint['initialization']['source_sha256'], record['source_sha256'])
            with self.assertRaisesRegex(ValueError, 'base_channels'):
                occlusion._initialize_weights(occlusion._model(occlusion.read_config({**self.config, 'base_channels': 4})),
                                              occlusion.read_config({**self.config, 'base_channels': 4}), source_path)

    def test_loss_skips_empty_hidden_region_and_rejects_nonfinite_valid(self):
        from prediction.occlusion import _loss_terms
        predicted = torch.zeros((1, 2, 1, 2, 2), requires_grad=True)
        target = torch.ones_like(predicted)
        valid = torch.ones_like(predicted, dtype=torch.bool)
        hidden = torch.zeros((1, 1, 2, 2), dtype=torch.bool)
        terms = _loss_terms(predicted, target, valid, hidden)
        self.assertEqual(terms['hidden'][1], 0)
        (terms['current'][0] / terms['current'][1] + terms['future'][0] / terms['future'][1]).backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        target[0, 0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            _loss_terms(predicted.detach(), target, valid, hidden)

    def test_invalid_config_is_rejected(self):
        from prediction.occlusion import read_config
        for bad in ({'horizon': 0}, {'kernel_size': 2}, {'seed': -1}, {'extra': 1}, {'min_valid_fraction': 0},
                    {'hidden_weight': float('nan')}, {'occlusion_shape': 'unknown'},
                    {'cache_frames': 1}, {'train_window_stride': 0}, {'train_window_stride': True},
                    {'gaussian_step_interval': 0}, {'gaussian_step_interval': True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                read_config(bad)
        self.assertEqual(read_config({})['gaussian_step_interval'], 1)

    def test_real_recordings_flow_from_training_to_evaluation(self):
        import json
        from prediction import occlusion
        from surface_mapper.camera import PointCloudFrame
        from surface_mapper.config import AppConfig, CameraConfig, GridConfig
        from surface_mapper.heightmap import HeightMap
        from surface_mapper.recording import SessionWriter

        geometry = AppConfig('camera', CameraConfig(fps=100),
                             GridConfig(0, 5, 0, 3, 1, 0, 1000), np.eye(4))
        for name in ('train', 'validation', 'test'):
            writer = SessionWriter(self.root / name, geometry, {'name': 'synthetic fixture'})
            writer.start()
            for i in range(4):
                height = np.full((3, 5), 100 + i, np.float32)
                valid = np.ones((3, 5), dtype=bool)
                frame = PointCloudFrame(i, i * 10., (i + 1) * 10_000_000,
                                        'test_clock', np.empty((0, 3), dtype=np.float32))
                writer.write(frame, HeightMap(height, valid, valid.astype(np.uint32), 15, 15))
            writer.finish()
        splits = self.root / 'splits.json'
        splits.write_text(json.dumps({name: [name] for name in ('train', 'validation', 'test')}))
        run = occlusion.train(splits, {**self.config, 'cache_frames': True, 'train_window_stride': 2},
                              self.root / 'actual_run', 'cpu')
        self.assertEqual(run['dataset_summaries']['train']['windows'], 1)
        self.assertEqual(run['dataset_summaries']['validation']['windows'], 2)
        self.assertEqual(run['dataset_summaries']['validation']['window_stride'], 1)
        report = occlusion.evaluate(run['checkpoint'], splits, self.root / 'actual_eval', device='cpu')
        self.assertEqual(report['windows'], 2)
        self.assertEqual(report['dataset_summary']['window_stride'], 1)
        self.assertGreater(report['leads'][1]['regions']['hidden']['target_valid'], 0)
        self.assertTrue((self.root / 'actual_eval/metrics.json').is_file())
        from prediction.occlusion_inference import replay
        replayed = replay(run['checkpoint'], self.root / 'test', self.root / 'replay', 'cpu')
        self.assertTrue(replayed['complete'])
        self.assertEqual(replayed['forecast_count'], 3)
        with np.load(self.root / 'replay/prediction_000002.npz') as forecast:
            self.assertEqual(forecast['prediction_mm'].shape, (2, 3, 5))
            np.testing.assert_array_equal(forecast['nominal_target_timestamps_ms'], [30., 40.])
            self.assertTrue(forecast['observation_valid'].all())
            self.assertTrue(forecast['history_observed'].all())


if __name__ == '__main__':
    unittest.main()
