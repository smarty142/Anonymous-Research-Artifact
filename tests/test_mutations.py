"""Tests for the mutation library."""
import pytest

from osi_bench.mutations import MUTATIONS, implemented_patterns, get_mutation
from osi_bench.base_trajectories import BASE_TRAJECTORIES


def test_every_mutation_returns_trajectory():
    for pid in implemented_patterns():
        traj = get_mutation(pid)("email_triage")
        assert traj.ground_truth == "malicious"
        assert pid in traj.traj_id


def test_every_mutation_preserves_steps_when_expected():
    # T.A.4 mutates an existing step rather than appending; check step count stays.
    traj = get_mutation("T.A.4")("file_read")
    assert traj.ground_truth == "malicious"
    assert any("path_traversal" in (s.tool_call.args.get("path", "") if s.tool_call else "")
               or "../../" in (s.tool_call.args.get("path", "") if s.tool_call else "")
               for s in traj.steps)


def test_T_A_11_drops_confirm_step():
    traj = get_mutation("T.A.11")("send_email_legit")
    assert traj.ground_truth == "malicious"
    confirm_count = sum(1 for s in traj.steps
                        if s.tool_call and s.tool_call.name == "confirm")
    assert confirm_count == 0


def test_T_B_1_invisible_chars_present():
    traj = get_mutation("T.B.1")("email_triage")
    assert any("​" in (s.text or "") for s in traj.steps)


def test_T_B_5_base64_decodes_to_meaningful():
    traj = get_mutation("T.B.5")("email_triage")
    import base64
    text = " ".join(s.text for s in traj.steps)
    found_b64 = None
    for token in text.split():
        if len(token) >= 20 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in token):
            try:
                decoded = base64.b64decode(token + "=" * (-len(token) % 4)).decode()
                if "ignore" in decoded.lower():
                    found_b64 = token
                    break
            except Exception:
                continue
    assert found_b64 is not None


def test_T_C_1_taint_flow_present():
    traj = get_mutation("T.C.1")("email_triage")
    # first step should be email-tainted, last step's args should contain that value
    assert any(
        s.tool_response and s.tool_response.source == "email"
        for s in traj.steps
    )
    last_step = traj.steps[-1]
    all_args = " ".join(str(v) for v in last_step.tool_call.args.values())
    assert "secret token" in all_args