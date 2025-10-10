"""Quality-focused tests for the SymPLONK security utilities."""

import json
import math
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")

from cryptography.plonk import plonk_security_notes as plonk


def test_ensure_safe_path_rejects_outside_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Paths escaping the configured base directory should be rejected."""

    monkeypatch.setattr(plonk, "BASE_DIR", tmp_path)

    safe = plonk._ensure_safe_path("nested/file.json")
    assert safe == tmp_path / "nested" / "file.json"

    with pytest.raises(ValueError):
        plonk._ensure_safe_path(Path("/etc/passwd"))

    with pytest.raises(TypeError):
        plonk._ensure_safe_path(123)  # type: ignore[arg-type]


def test_validate_config_data_enforces_positive_flow_time() -> None:
    """flow_time must be a positive floating point number when provided."""

    valid = plonk._validate_config_data({"n": 8, "p": 97, "flow_time": math.pi / 2})
    assert math.isclose(valid["flow_time"], math.pi / 2)

    with pytest.raises(ValueError):
        plonk._validate_config_data({"n": 8, "p": 97, "flow_time": 0})

    with pytest.raises(ValueError):
        plonk._validate_config_data({"n": 8, "p": 97, "flow_time": -1})


def test_commitment_from_dict_requires_complete_consistent_data() -> None:
    """Incomplete or inconsistent commitment payloads must be rejected."""

    commitment = plonk.Commitment(
        transformed_evaluations=np.array([1 + 1j, 2 + 0.5j]),
        flow_time=math.pi / 4,
        secret_original_evaluations=np.array([1 + 1j, 2 + 0.5j]),
        r=complex(0.3, -0.1),
        mask=np.array([0.5 + 0.2j, -0.4 + 0.1j]),
    )

    payload = commitment.to_dict()
    reconstructed = plonk.Commitment.from_dict(payload)
    assert np.allclose(reconstructed.transformed_evaluations, commitment.transformed_evaluations)

    payload.pop("mask")
    with pytest.raises(ValueError):
        plonk.Commitment.from_dict(payload)

    payload["mask"] = {"real": [0.1, 0.2], "imag": [0.3, 0.4]}
    payload["secret_original_evaluations"] = {"real": [0.0], "imag": [0.0]}
    with pytest.raises(ValueError):
        plonk.Commitment.from_dict(payload)


def test_symplonk_round_trip_with_persistence(tmp_path: Path) -> None:
    """End-to-end SymPLONK workflow should succeed on CPU and persist commitments."""

    symplonk = plonk.SymPLONK(n=8, p=97, epsilon=1e-8, use_gpu=False)
    secret = [3, 1, 4]
    flow_time = math.pi / 6

    commitment = symplonk.alice_prove(secret, flow_time)
    assert isinstance(commitment, plonk.Commitment)
    assert symplonk.bob_verify(commitment)

    commitment_path = tmp_path / "commitment.json"
    symplonk.save_commitment(commitment, commitment_path)
    loaded_commitment = symplonk.load_commitment(commitment_path)
    assert symplonk.bob_verify(loaded_commitment)

    with commitment_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    raw_data["mask"]["real"].append(0.0)
    corrupted_path = tmp_path / "corrupted.json"
    corrupted_path.write_text(json.dumps(raw_data), encoding="utf-8")

    with pytest.raises(ValueError):
        symplonk.load_commitment(corrupted_path)
