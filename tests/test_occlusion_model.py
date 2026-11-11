"""Causal and shape contracts for the occlusion-aware model."""

import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    import torch
    from prediction.occlusion_model import OcclusionUNetConvLSTM


@unittest.skipUnless(TORCH_AVAILABLE, 'optional prediction dependencies are not installed')
class OcclusionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def inputs(batch=2, time=3, height=5, width=7):
        x = torch.zeros(batch, time, 4, height, width)
        x[:, :, 0] = torch.randn(batch, time, height, width)
        x[:, :, 1] = 1
        return x

    def test_shapes_rollout_and_gradients_for_odd_and_tiny_grids(self):
        model = OcclusionUNetConvLSTM(base_channels=2)
        for height, width in ((5, 7), (1, 1), (1, 3)):
            inputs = self.inputs(height=height, width=width).requires_grad_()
            output = model(inputs, rollout_steps=2)
            self.assertEqual(output.shape, (2, 3, 1, height, width))
            output.square().sum().backward()
            self.assertTrue(torch.isfinite(inputs.grad).all())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_unobserved_frame_matches_open_loop_propagation(self):
        torch.manual_seed(17)
        model = OcclusionUNetConvLSTM(base_channels=2).eval()
        history = self.inputs()
        missing = torch.zeros_like(history[:, :1])
        missing[:, :, 0] = 1000  # a placeholder is not a measurement
        missing[:, :, 2:] = 1
        with torch.no_grad():
            predicted = model(history, 1)
            later = model(torch.cat((history, missing), dim=1), 0)
        torch.testing.assert_close(predicted[:, 1], later[:, 0])

    def test_missing_pixel_values_do_not_change_predictions(self):
        model = OcclusionUNetConvLSTM(base_channels=2).eval()
        inputs = self.inputs()
        inputs[:, :, 1, 1:3, 2:4] = 0
        changed = inputs.clone()
        changed[:, :, 0, 1:3, 2:4] = 999
        with torch.no_grad():
            torch.testing.assert_close(model(inputs, 2), model(changed, 2), rtol=0, atol=0)

    def test_invalid_contract_is_rejected(self):
        model = OcclusionUNetConvLSTM(base_channels=2)
        for value in (0, -1, 2, 2.5, True):
            with self.assertRaises(ValueError):
                OcclusionUNetConvLSTM(base_channels=2, kernel_size=value)
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                model(self.inputs(), value)
        for channel, value in ((1, .5), (2, 1.5), (3, .5), (0, float('nan'))):
            inputs = self.inputs()
            inputs[0, 0, channel, 0, 0] = value
            with self.assertRaises(ValueError):
                model(inputs, 1)

    def test_default_model_keeps_paper_parameters(self):
        model = OcclusionUNetConvLSTM()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 708_753)
        self.assertTrue(any(k.startswith('dynamics.') for k in model.state_dict()))
        self.assertTrue(any(k.startswith('correction.') for k in model.state_dict()))

    def test_ablation_parameter_counts(self):
        expected = {
            (('depth', 'mask', 'age'), 'dynamics_correction'): 708_753 - 144,
            (('depth', 'mask'), 'dynamics_correction'): 708_753 - 2 * 144,
            (('depth',), 'dynamics_correction'): 708_753 - 3 * 144,
            (('depth', 'mask', 'age', 'never'), 'stacked'): 708_753,
            (('depth', 'mask', 'age', 'never'), 'single'): 708_753 - 295_168,
        }
        for (channels, recurrence), count in expected.items():
            model = OcclusionUNetConvLSTM(input_channels=channels, recurrence=recurrence)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count, (channels, recurrence))

    def test_dropped_channels_do_not_change_predictions(self):
        torch.manual_seed(3)
        model = OcclusionUNetConvLSTM(base_channels=2, input_channels=('mask', 'depth')).eval()
        self.assertEqual(model.input_channels, ('depth', 'mask'))
        inputs = self.inputs()
        inputs[:, :, 1, 1:3, 2:4] = 0
        changed = inputs.clone()
        changed[:, :, 2, 1:3, 2:4] = 1
        changed[:, :, 3, 1:3, 2:4] = 1
        with torch.no_grad():
            torch.testing.assert_close(model(inputs, 2), model(changed, 2), rtol=0, atol=0)

    def test_ablated_recurrences_follow_the_same_contract(self):
        for recurrence in ('single', 'stacked'):
            torch.manual_seed(5)
            model = OcclusionUNetConvLSTM(base_channels=2, recurrence=recurrence).eval()
            history = self.inputs()
            missing = torch.zeros_like(history[:, :1])
            missing[:, :, 0] = 1000
            missing[:, :, 2:] = 1
            with torch.no_grad():
                predicted = model(history, 2)
                later = model(torch.cat((history, missing), dim=1), 0)
            self.assertEqual(predicted.shape, (2, 3, 1, 5, 7))
            torch.testing.assert_close(predicted[:, 1], later[:, 0])

    def test_invalid_ablation_settings_are_rejected(self):
        for channels in ((), ('mask',), ('depth', 'depth'), ('depth', 'height'), 'depth'):
            with self.assertRaises(ValueError):
                OcclusionUNetConvLSTM(base_channels=2, input_channels=channels)
        with self.assertRaises(ValueError):
            OcclusionUNetConvLSTM(base_channels=2, recurrence='gru')


if __name__ == '__main__':
    unittest.main()
