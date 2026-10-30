"""Exercise offline dependency preparation with a real, pinned SLEEF checkout."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def dependency(tmp_path):
    location = os.environ.get("QWEN_SLEEF_SOURCE_DIR")
    if not location:
        pytest.skip("Set QWEN_SLEEF_SOURCE_DIR to an unmodified SLEEF 3.8 tree")
    source = tmp_path / "upstream"
    shutil.copytree(location, source)
    module = Path(__file__).resolve().parents[3] / "native/arm_rope_trig_runtime/Sleef.cmake"
    project = tmp_path / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.20)\nproject(check NONE)\n'
        f'include("{module.as_posix()}")\nqwen_prepare_sleef(prepared)\n')
    build = tmp_path / "build"

    def configure(use_environment=False):
        options = [] if use_environment else [f"-DQWEN_SLEEF_SOURCE_DIR={source}"]
        return subprocess.run(
            ["cmake", "-S", str(project), "-B", str(build), *options],
            env=dict(os.environ, HTTPS_PROXY="http://127.0.0.1:1",
                     HTTP_PROXY="http://127.0.0.1:1", ALL_PROXY="http://127.0.0.1:1",
                     QWEN_SLEEF_SOURCE_DIR=str(source)),
            capture_output=True, text=True)

    return source, build / "sleef-rope", configure


def test_environment_selects_offline_source(dependency):
    _, prepared, configure = dependency
    result = configure(use_environment=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (prepared / "helperpurec.h").is_file()


def test_offline_preparation_preserves_source_and_is_repeatable(dependency):
    source, prepared, configure = dependency
    helper = source / "src/arch/helperpurec.h"
    original = helper.read_bytes()
    result = configure()
    assert result.returncode == 0, result.stdout + result.stderr
    output = prepared / "helperpurec.h"
    assert hashlib.sha256(output.read_bytes()).hexdigest() == (
        "1dec369567711e9cf902d64d2457a3d9d1b757c2a83db5f14745d8161dabe7ac")
    timestamp = output.stat().st_mtime_ns
    result = configure()
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.stat().st_mtime_ns == timestamp
    assert helper.read_bytes() == original


@pytest.mark.parametrize("relative", ["src/arch/helperpurec.h", "src/libm/sleefsimdsp.c"])
def test_changed_upstream_source_is_rejected(dependency, relative):
    source, _, configure = dependency
    with (source / relative).open("ab") as stream:
        stream.write(b"\n/* changed dependency */\n")
    result = configure()
    assert result.returncode != 0
    assert f"SLEEF 3.8 SHA256 mismatch: {relative}" in result.stderr
