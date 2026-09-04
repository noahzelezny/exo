"""The one-knob rule: EXO_MTP alone selects the sequential engine, but only
when drafting would actually engage (registered family + sidecar on disk).
A non-drafting model must stay on the batch engine — routing it through
sequential is a measured 4x decode loss (docs/DECODE-LOOP-OVERHEAD.md)."""

from exo.worker.engines.mlx import builder as builder_mod
from exo.worker.engines.mlx.builder import _mtp_would_engage


class _Spec:
    sidecar_name = "mtp-head-q6.safetensors"


def _wire(monkeypatch, tmp_path, *, resolves: bool, sidecar: bool):
    if sidecar:
        (tmp_path / _Spec.sidecar_name).write_bytes(b"x")

    def fake_resolve(model):
        if not resolves:
            raise KeyError("no MTP family registered")
        return _Spec()

    import exo.download.download_utils as dl
    import exo.worker.engines.mlx.mtp.registry as reg

    monkeypatch.setattr(reg, "resolve", fake_resolve)
    monkeypatch.setattr(dl, "build_model_path", lambda _id: tmp_path)


def test_unset_env_never_engages(monkeypatch, tmp_path):
    monkeypatch.delenv("EXO_MTP", raising=False)
    _wire(monkeypatch, tmp_path, resolves=True, sidecar=True)
    assert _mtp_would_engage(object(), "o/m") is False


def test_unregistered_family_stays_on_batch(monkeypatch, tmp_path):
    monkeypatch.setenv("EXO_MTP", "1")
    _wire(monkeypatch, tmp_path, resolves=False, sidecar=True)
    assert _mtp_would_engage(object(), "o/m") is False


def test_missing_sidecar_stays_on_batch(monkeypatch, tmp_path):
    monkeypatch.setenv("EXO_MTP", "1")
    _wire(monkeypatch, tmp_path, resolves=True, sidecar=False)
    assert _mtp_would_engage(object(), "o/m") is False


def test_drafting_model_selects_sequential(monkeypatch, tmp_path):
    monkeypatch.setenv("EXO_MTP", "1")
    _wire(monkeypatch, tmp_path, resolves=True, sidecar=True)
    assert _mtp_would_engage(object(), "o/m") is True
