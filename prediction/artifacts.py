"""Prediction artifacts shared by NumPy baselines and learned models."""

import json
from pathlib import Path

import numpy as np


def prepare_output(path) -> Path:
    output = Path(path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'output must be a new or empty directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path, payload) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class ForecastWriter:
    def __init__(self, output, predictor, mode):
        self.output = prepare_output(output)
        self.predictor = predictor
        self.count = 0
        self.manifest = {
            'schema_version': 1, 'complete': False, 'mode': mode,
            'config': predictor.config.to_dict(), 'geometry': predictor.geometry,
            'forecast_count': 0,
            'mask_meaning': ('observation_valid and history_observed describe past measurements; '
                             'baseline prediction_valid describes fit support. None is calibrated confidence.'),
            'timestamps': 'source and target timestamps remain in the source clock domain',
        }
        write_json(self.output / 'manifest.json', self.manifest)
        self.log = (self.output / 'forecasts.jsonl').open('w', encoding='utf-8')

    def write(self, forecast, source_frame_number, **extra):
        filename = f'prediction_{self.count:06d}.npz'
        metadata = {
            'file': filename, 'source_frame_number': int(source_frame_number),
            'source_timestamp_ms': forecast.source_timestamp_ms,
            'target_timestamp_ms': forecast.target_timestamp_ms,
            'timestamp_domain': forecast.timestamp_domain,
            'inference_ms': forecast.inference_ms, **extra,
        }
        arrays = {'prediction_mm': forecast.prediction_mm,
                  'observation_valid': forecast.observation_valid,
                  'history_observed': forecast.history_observed}
        for name in ('prediction_valid', 'velocity_mm_s', 'acceleration_mm_s2'):
            value = getattr(forecast, name, None)
            if value is not None:
                arrays[name] = value
        if 'prediction_valid' in arrays:
            metadata['prediction_valid_fraction'] = float(arrays['prediction_valid'].mean())
        temporary = self.output / ('.' + filename + '.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays,
                                **{key: value for key, value in metadata.items() if key != 'file'})
        temporary.replace(self.output / filename)
        self.log.write(json.dumps(metadata, allow_nan=False) + '\n')
        self.log.flush()
        self.count += 1

    def close(self, complete):
        self.log.close()
        self.manifest.update(complete=complete, forecast_count=self.count,
                             buffer_resets=self.predictor.reset_count,
                             last_reset_reason=self.predictor.last_reset_reason)
        write_json(self.output / 'manifest.json', self.manifest)
