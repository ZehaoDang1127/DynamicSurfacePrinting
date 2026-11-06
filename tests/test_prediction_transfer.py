"""Transfer changes the training initialization, not the target data contract."""

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest('install requirements-prediction.txt to test transfer learning') from exc

from prediction.cli import build_parser, run
from prediction.config import PredictionConfig
from prediction.data import geometry_signature
from prediction.model import ConvLSTMPredictor
from prediction import training
from surface_mapper.camera import PointCloudFrame
from surface_mapper.config import AppConfig, CameraConfig, GridConfig
from surface_mapper.heightmap import HeightMap
from surface_mapper.recording import SessionWriter


class TransferTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.source_config = PredictionConfig(history=3, horizon=1, expected_dt_ms=200,
                                              hidden_channels=(2, 3), epochs=1)
        self.target_config = replace(self.source_config, history=4, horizon=2,
                                     expected_dt_ms=100, batch_size=2, learning_rate=.002)
        self.source = self.root / 'source.pt'
        source_camera = AppConfig('tds_camera', CameraConfig(fps=5),
                                  GridConfig(-20, 20, -20, 20, 20, 400, 2000), np.eye(4))
        source_model = ConvLSTMPredictor(self.source_config)
        with torch.no_grad():
            for index, parameter in enumerate(source_model.parameters()):
                parameter.fill_(.01 + index * .001)
        self.source_state = {k: v.detach().clone() for k, v in source_model.state_dict().items()}
        self.source_checkpoint = {
            'schema_version': 1, 'state_dict': self.source_state,
            'config': self.source_config.to_dict(),
            'normalization': {'mean_mm': 1000., 'scale_mm': 80.},
            'geometry': geometry_signature(source_camera.to_dict()), 'shape': [2, 2],
            'train_sessions': ['source/train'], 'validation_sessions': ['source/validation'],
            'test_sessions': ['source/test'],
            # Extraneous optimizer data must never become a resumed target optimizer.
            'optimizer_state': {'state': {'old': 'must not be restored'}},
        }
        torch.save(self.source_checkpoint, self.source)
        transform = np.eye(4)
        transform[:3, 3] = [170., 83., 150.]
        self.target_camera = AppConfig('printer', CameraConfig(fps=10),
                                      GridConfig(170, 178, 80, 86, 2, 140, 200), transform)
        self.training_values = []
        for name, base in (('train', 150.), ('validation', 170.), ('test', 180.)):
            writer = SessionWriter(self.root / name, self.target_camera,
                                   {'name': 'synthetic target test fixture'})
            writer.start()
            for index in range(9):
                height = (base + index + np.arange(12).reshape(3, 4) * .1).astype(np.float32)
                valid = np.ones((3, 4), dtype=bool)
                frame = PointCloudFrame(index, index * 100., (index + 1) * 100_000_000,
                                        'test_clock', np.empty((0, 3), dtype=np.float32))
                writer.write(frame, HeightMap(height, valid, valid.astype(np.uint32), 12, 12))
                if name == 'train':
                    self.training_values.extend(height.ravel().astype(float))
            writer.finish()
        self.splits = self.root / 'splits.json'
        self.splits.write_text(json.dumps({name: [name] for name in ('train', 'validation', 'test')}))

    def test_target_training_uses_source_weights_but_new_geometry_normalization_and_optimizer(self):
        original_epoch = training._run_epoch
        observed_initial_state = []

        def inspect_first_epoch(model, loader, device, scale_mm, optimizer=None, grad_clip=1.):
            if optimizer is not None and not observed_initial_state:
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor.cpu(), self.source_state[name], rtol=0, atol=0)
                self.assertEqual(len(optimizer.state), 0)
                self.assertEqual(optimizer.param_groups[0]['lr'], self.target_config.learning_rate)
                observed_initial_state.append(True)
            return original_epoch(model, loader, device, scale_mm, optimizer, grad_clip)

        source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        with patch('prediction.training._run_epoch', side_effect=inspect_first_epoch), redirect_stdout(io.StringIO()):
            report = training.train(self.splits, self.target_config, self.root / 'trained', 'cpu',
                                    initialize_from=self.source)
        self.assertEqual(observed_initial_state, [True])
        model, config, normalization, checkpoint = training.load_checkpoint(self.root / 'trained/best.pt', 'cpu')
        self.assertEqual(config, self.target_config)
        self.assertEqual(checkpoint['geometry'], geometry_signature(self.target_camera.to_dict()))
        self.assertEqual(checkpoint['shape'], [3, 4])
        self.assertAlmostEqual(normalization.mean_mm, float(np.mean(self.training_values)), places=6)
        self.assertAlmostEqual(normalization.scale_mm, float(np.std(self.training_values)), places=6)
        self.assertNotEqual(normalization.to_dict(), self.source_checkpoint['normalization'])
        self.assertEqual(model(torch.zeros(1, 4, 2, 3, 4)).shape, (1, 1, 3, 4))
        self.assertTrue(any(not torch.equal(tensor.cpu(), self.source_state[name])
                            for name, tensor in model.state_dict().items()))
        init = checkpoint['initialization']
        self.assertEqual(init['source_sha256'], source_hash)
        self.assertEqual(init['source_geometry'], self.source_checkpoint['geometry'])
        self.assertEqual(init['source_config'], self.source_config.to_dict())
        self.assertFalse(init['optimizer_state_transferred'])
        self.assertNotIn('optimizer_state', checkpoint)
        saved_run = json.loads((self.root / 'trained/run.json').read_text())
        self.assertEqual(saved_run['initialization'], json.loads(json.dumps(init)))
        self.assertEqual(report['initialization'], init)
        self.assertEqual(checkpoint['train_sessions'], [str((self.root / 'train').resolve())])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), source_hash)

    def test_architecture_mismatches_are_rejected_before_training(self):
        for changes in ({'hidden_channels': (3, 3)}, {'kernel_size': 5}):
            with self.subTest(changes=changes):
                target = replace(self.target_config, **changes)
                with patch('prediction.training._run_epoch') as epoch:
                    with self.assertRaisesRegex(ValueError, 'initialization architecture differs'):
                        training.train(self.splits, target, self.root / 'rejected', 'cpu',
                                       initialize_from=self.source)
                    epoch.assert_not_called()
                self.assertFalse((self.root / 'rejected').exists())

    def test_nonfinite_initialization_is_rejected(self):
        checkpoint = dict(self.source_checkpoint)
        checkpoint['state_dict'] = {k: v.clone() for k, v in self.source_state.items()}
        next(iter(checkpoint['state_dict'].values())).flatten()[0] = float('nan')
        bad_source = self.root / 'nonfinite.pt'
        torch.save(checkpoint, bad_source)
        with self.assertRaisesRegex(ValueError, 'nonfinite model parameters'):
            training.train(self.splits, self.target_config, self.root / 'rejected', 'cpu',
                           initialize_from=bad_source)

    def test_cli_passes_initialization_and_defaults_to_training_from_scratch(self):
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(self.target_config.to_dict()))
        command = ['train', '--splits', str(self.splits), '--config', str(config_path),
                   '--output', str(self.root / 'cli_run'), '--device', 'cpu', '--threads', '1']
        self.assertIsNone(build_parser().parse_args(command).initialize_from)
        args = build_parser().parse_args(command + ['--initialize-from', str(self.source)])
        with patch('prediction.training.train', return_value={}) as trainer, redirect_stdout(io.StringIO()):
            self.assertEqual(run(args), 0)
        self.assertEqual(trainer.call_args.kwargs['initialize_from'], self.source)
        del args.initialize_from  # Preserve callers constructing the old argument namespace.
        with patch('prediction.training.train', return_value={}) as trainer, redirect_stdout(io.StringIO()):
            self.assertEqual(run(args), 0)
        self.assertIsNone(trainer.call_args.kwargs['initialize_from'])


if __name__ == '__main__':
    unittest.main()
