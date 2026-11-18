"""Model-specific inputs and Slurm submission for unconfounded experiments."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dev.schema_llm.explore.stop_steering import unconfounded_data as data
from dev.schema_llm.explore.stop_steering import unconfounded_generate as generate

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    ("model", "expected"),
    [("phi4mr", 84), ("qwen34b", 118)],
)
def test_model_population_vectors_and_trace_coordinates(monkeypatch, model, expected):
    monkeypatch.setenv("SCHEMA_HEAD_FV_MODEL", model)
    if not (ROOT / data.population_path()).exists():
        pytest.skip("model-specific population not present")
    population = data.load_population(ROOT)
    assert len(population) == expected
    tokens = data.load_tokens(ROOT)
    for _, target in population.iterrows():
        data.validate_trace(data.build_trace(tokens, target["item_id"]), target)
    supplied = data.load_vectors(ROOT)
    for arm in ("stop", "shuffled"):
        vector, dose_unit = generate.arm_vector_and_dose_unit(ROOT, arm, supplied)
        assert np.isclose(np.linalg.norm(vector), 1)
        assert dose_unit > 0


def test_vector_rejects_wrong_model_for_both_arms(monkeypatch):
    monkeypatch.setenv("SCHEMA_HEAD_FV_MODEL", "qwen34b")
    supplied = data.load_vectors(ROOT)
    supplied["model"] = np.asarray("phi4mr")
    for arm in ("stop", "shuffled"):
        with pytest.raises(ValueError, match="model is not qwen34b"):
            generate.arm_vector_and_dose_unit(ROOT, arm, supplied)


def test_rejects_incorrect_solved_coordinates(monkeypatch):
    monkeypatch.setenv("SCHEMA_HEAD_FV_MODEL", "qwen34b")
    target = data.load_population(ROOT).iloc[0].copy()
    trace = data.build_trace(data.load_tokens(ROOT), target["item_id"])
    target["solved_position_abs"] += 1
    with pytest.raises(ValueError, match="solved coordinates disagree"):
        data.validate_trace(trace, target)


def test_vector_population_size_without_release_manifest(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SCHEMA_HEAD_FV_MODEL", "qwen34b")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").symlink_to(ROOT / "data", target_is_directory=True)
    monkeypatch.setattr("sys.argv", ["unconfounded_generate", "--root", str(tmp_path),
                                    "--population-size"])
    generate.main()
    assert capsys.readouterr().out.strip() == "118"


def test_vector_resume_ignores_only_legacy_release_hash(tmp_path):
    config = {"population_sha256": "population", "vectors_sha256": "vector", "budget": -1}
    generate._bind_generation_config(tmp_path, config)
    state = tmp_path / "generation.json"
    assert json.loads(state.read_text()) == config
    legacy = {**config, "release": "old-release"}
    state.write_text(json.dumps(legacy))
    generate._bind_generation_config(tmp_path, config)
    assert json.loads(state.read_text()) == legacy
    for changed in ("population_sha256", "vectors_sha256", "budget"):
        with pytest.raises(ValueError, match="generation inputs/settings changed"):
            generate._bind_generation_config(tmp_path, {**config, changed: "different"})


@pytest.mark.parametrize(
    ("script", "output_env", "array_script", "merge_script"),
    [
        ("submit_stop_unconfounded_qwen.sh", "STOP_OUT",
         "stop_unconfounded_generate_array.sbatch", "stop_unconfounded_generate_merge.sbatch"),
        ("submit_stop_unconfounded_prompt_qwen.sh", "STOP_PROMPT_OUT",
         "stop_unconfounded_prompt_array.sbatch", "stop_unconfounded_prompt_merge.sbatch"),
    ],
)
def test_qwen_submitters_export_model_and_dependency(
    tmp_path, script, output_env, array_script, merge_script
):
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(
        '#!/usr/bin/env bash\n'
        'printf "model=%s\\n" "$SCHEMA_HEAD_FV_MODEL" >> "$CALLS"\n'
        'printf "%s\\n" "$@" >> "$CALLS"\n'
        'echo 12345\n'
    )
    sbatch.chmod(0o755)
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALLS": str(calls),
        output_env: str(tmp_path / "output"),
    }
    subprocess.run(["bash", str(ROOT / "scripts/slurm" / script)], env=env, check=True,
                   capture_output=True, text=True)
    log = calls.read_text()
    assert log.count("model=qwen34b") == 2
    assert f"{output_env}={tmp_path / 'output'}" in log
    assert "--dependency=afterok:12345" in log
    assert array_script in log and merge_script in log
