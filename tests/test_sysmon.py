import numpy as np
import pytest
import torch

from sysmon.collector import MetricCollector
from sysmon.data import Scaler, WindowDataset
from sysmon.detector import AnomalyDetector, evaluate
from sysmon.features import FEATURES
from sysmon.model import LSTMAutoencoder
from sysmon.synthetic import generate
from sysmon.train import TrainConfig, train


def test_collector_returns_every_feature():
    sample = MetricCollector().sample()
    assert list(sample) == FEATURES
    assert all(v >= 0 for v in sample.values())


def test_scaler_standardises_and_roundtrips():
    x, _ = generate(2000, seed=0)
    s = Scaler.fit(x)
    z = s.transform(x)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-3)
    s2 = Scaler.from_state_dict(s.state_dict())
    np.testing.assert_allclose(s2.transform(x), z)


def test_window_dataset_shapes():
    ds = WindowDataset(np.zeros((100, len(FEATURES)), np.float32), window=10, stride=5)
    assert len(ds) == 19
    assert ds[0].shape == (10, len(FEATURES))
    with pytest.raises(ValueError):
        WindowDataset(np.zeros((5, len(FEATURES)), np.float32), window=10)


def test_model_reconstructs_input_shape():
    m = LSTMAutoencoder(len(FEATURES), hidden=8, latent=4)
    x = torch.randn(3, 12, len(FEATURES))
    assert m(x).shape == x.shape
    assert m.feature_errors(x).shape == (3, len(FEATURES))


def test_evaluate_uses_window_labels():
    labels = np.array([0, 0, 1, 0, 0, 0])
    scores = np.array([np.nan, 0.0, 1.0, 1.0, 0.0, 0.0])  # window=2: step 3 still contains the incident
    m = evaluate(scores, labels, threshold=0.5, window=2)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 0, 0)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    x, _ = generate(3000, seed=1)
    path = tmp_path_factory.mktemp("m") / "m.pt"
    train(x, TrainConfig(window=20, hidden=32, latent=8, epochs=8), path, log=lambda *_: None)
    return path


def test_end_to_end_detects_injected_anomalies(checkpoint):
    x, y = generate(3000, anomaly_rate=0.08, seed=2)
    det = AnomalyDetector(checkpoint)
    scores = det.score_series(x)
    assert np.isnan(scores[: det.window - 1]).all()
    m = evaluate(scores, y, det.threshold, det.window)
    assert m["f1"] > 0.8, m


def test_streaming_matches_batch_scoring(checkpoint):
    x, _ = generate(60, seed=3)
    det = AnomalyDetector(checkpoint, min_consecutive=1)
    batch = det.score_series(x)
    streamed = [det.update(dict(zip(FEATURES, row))) for row in x]
    assert all(v is None for v in streamed[: det.window - 1])
    np.testing.assert_allclose([v.score for v in streamed[det.window - 1 :]], batch[det.window - 1 :], rtol=1e-4)
    assert streamed[-1].top_features[0][0] in FEATURES
