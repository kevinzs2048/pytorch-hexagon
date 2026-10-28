"""Keep requested model policy separate from the backend's legacy attributes."""
from types import SimpleNamespace

import pytest

from qwen3.runtime.arm.compile_qwen3_artifact import codegen_options


@pytest.mark.parametrize("enabled", [True, False])
def test_process_policy_preserves_request_and_avoids_private_attribute(enabled):
    calls = []
    backend = SimpleNamespace(configure_process_geninsert=calls.append)
    requested = {"enableHexagonGenInsert": str(enabled), "other": "unchanged"}
    effective, policy = codegen_options(backend, requested, "process")
    assert calls == [enabled]
    assert requested["enableHexagonGenInsert"] == str(enabled)
    assert effective == {"enableHexagonGenInsert": "True", "other": "unchanged"}
    assert policy == {"scope": "isolated_process", "llvm_option": "hexagon-insert",
                      "enabled": enabled}


def test_process_policy_requires_matching_compiler_instead_of_silent_fallback():
    with pytest.raises(ValueError, match="lacks the process"):
        codegen_options(object(), {"enableHexagonGenInsert": "False"}, "process")


def test_process_policy_rejects_malformed_boolean_before_setting_flags():
    with pytest.raises(ValueError, match="True or False"):
        codegen_options(object(), {"enableHexagonGenInsert": "false-ish"}, "process")


def test_legacy_policy_remains_available_for_comparison():
    requested = {"enableHexagonGenInsert": "False"}
    effective, policy = codegen_options(object(), requested, "function")
    assert effective == requested
    assert policy == {"scope": "function_attribute"}
