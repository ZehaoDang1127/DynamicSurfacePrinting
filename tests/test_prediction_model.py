"""Regression coverage for recurrent states, masking, and training provenance."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    import torch
    from prediction.config import PredictionConfig
    from prediction.model import ConvLSTMPredictor, masked_mae


@unittest.skipUnless(TORCH_AVAILABLE, 'optional prediction dependencies are not installed')
class ConvLSTMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def config(self, **kwargs):
        return PredictionConfig(history=3, hidden_channels=(3, 4), **kwargs)

    def test_odd_spatial_shape_and_backpropagation_to_first_frame(self) -> None:
        torch.manual_seed(8)
        model = ConvLSTMPredictor(self.config())
        inputs = torch.randn(2, 3, 2, 5, 7, requires_grad=True)
        output = model(inputs)
        self.assertEqual(output.shape, (2, 1, 5, 7))
        output.sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(float(inputs.grad[:, 0].abs().sum()), 0)
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_independent_windows_reset_hidden_state(self) -> None:
        model = ConvLSTMPredictor(self.config()).eval()
        inputs = torch.randn(1, 3, 2, 5, 7)
        with torch.no_grad():
            before = model(inputs)
            model(torch.randn_like(inputs) * 10)
            after = model(inputs)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_wrong_shape_and_history_are_rejected(self) -> None:
        model = ConvLSTMPredictor(self.config())
        for inputs in (torch.zeros(1, 3, 5, 7), torch.zeros(1, 3, 1, 5, 7), torch.zeros(1, 2, 2, 5, 7)):
            with self.assertRaises(ValueError):
                model(inputs)

    def test_mask_removes_invalid_nan_before_arithmetic(self) -> None:
        prediction = torch.tensor([2.0, float('nan'), 5.0], requires_grad=True)
        target = torch.tensor([1.0, float('nan'), 3.0])
        loss = masked_mae(prediction, target, torch.tensor([True, False, True]))
        self.assertEqual(float(loss.detach()), 1.5)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([0.5, 0.0, 0.5]))

    def test_empty_mask_is_differentiable_even_with_nan(self) -> None:
        prediction = torch.tensor([float('nan')], requires_grad=True)
        loss = masked_mae(prediction, prediction.detach(), torch.tensor([False]))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertEqual(float(prediction.grad[0]), 0.0)

    def test_valid_nonfinite_prediction_or_target_is_error(self) -> None:
        valid = torch.tensor([True])
        for prediction, target in ((float('nan'), 1.0), (1.0, float('nan')), (float('inf'), 1.0)):
            with self.assertRaises(ValueError):
                masked_mae(torch.tensor([prediction]), torch.tensor([target]), valid)

    def test_tiny_training_checkpoint_and_train_only_normalization(self) -> None:
        from prediction.data import Normalization
        from prediction.training import load_checkpoint, train

        config = self.config(epochs=2, patience=2, batch_size=2, seed=13)
        samples = [
            {
                'inputs': np.full((3, 2, 3, 5), 0.1 * index, np.float32),
                'target': np.full((1, 3, 5), 0.2, np.float32),
                'target_valid': np.ones((1, 3, 5), dtype=bool),
            }
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = {
                name: [SimpleNamespace(path=root / name, geometry={'frame_name': 'test'}, shape=(3, 5))]
                for name in ('train', 'validation', 'test')
            }
            normalization = Normalization(mean_mm=100.0, scale_mm=2.0)
            with (
                patch('prediction.training.load_splits', return_value=splits),
                patch('prediction.training.fit_normalization', return_value=normalization) as fit,
                patch('prediction.training.HeightSequenceDataset', return_value=samples) as dataset,
            ):
                report = train(root / 'splits.json', config, root / 'run', device='cpu')
                fit.assert_called_once_with(splits['train'])
                self.assertEqual(dataset.call_count, 2)
                self.assertEqual(dataset.call_args_list[0].args[0], splits['train'])
                self.assertEqual(dataset.call_args_list[1].args[0], splits['validation'])
            self.assertFalse(report['test_evaluated'])
            self.assertTrue(Path(report['history']).is_file())
            self.assertTrue(np.isfinite(report['best_validation_mae_mm']))
            loaded, loaded_config, loaded_norm, checkpoint = load_checkpoint(report['checkpoint'], 'cpu')
            self.assertEqual(config, loaded_config)
            self.assertEqual(loaded_norm.to_dict(), normalization.to_dict())
            self.assertFalse(loaded.training)
            self.assertEqual(checkpoint['test_sessions'], [str((root / 'test').resolve())])
            reference = ConvLSTMPredictor(config)
            reference.load_state_dict(checkpoint['state_dict'])
            batch = torch.as_tensor(samples[0]['inputs']).unsqueeze(0)
            with torch.no_grad():
                torch.testing.assert_close(loaded(batch), reference(batch))
            with self.assertRaises(FileExistsError):
                train(root / 'splits.json', config, root / 'run', device='cpu')

    def test_validation_aggregation_is_weighted_by_observed_pixels(self) -> None:
        from prediction.training import _run_epoch

        class IdentityModel(torch.nn.Module):
            def forward(self, values):
                return values

        batches = [
            {'inputs': torch.tensor([[[[1.0, 50.0]]]]), 'target': torch.zeros(1, 1, 1, 2), 'target_valid': torch.tensor([[[[True, False]]]])},
            {'inputs': torch.tensor([[[[3.0, 3.0]]]]), 'target': torch.zeros(1, 1, 1, 2), 'target_valid': torch.tensor([[[[True, True]]]])},
        ]
        mae, count = _run_epoch(IdentityModel(), batches, torch.device('cpu'), scale_mm=2.0)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(mae, 14.0 / 3.0)


    def test_interrupted_training_records_status(self) -> None:
        import json
        from prediction.data import Normalization
        from prediction.training import train

        config = self.config(epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = {
                name: [SimpleNamespace(path=root / name, geometry={}, shape=(3, 5))]
                for name in ('train', 'validation', 'test')
            }
            with (
                patch('prediction.training.load_splits', return_value=splits),
                patch('prediction.training.fit_normalization', return_value=Normalization(1.0, 1.0)),
                patch('prediction.training.HeightSequenceDataset', return_value=[{}]),
                patch('prediction.training._run_epoch', side_effect=KeyboardInterrupt),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    train(root / 'splits.json', config, root / 'run', device='cpu')
            metadata = json.loads((root / 'run' / 'run.json').read_text())
            self.assertEqual(metadata['status'], 'interrupted')
            self.assertEqual(metadata['epochs_completed'], 0)
            self.assertIn('finished_utc', metadata)

if __name__ == '__main__':
    unittest.main()
