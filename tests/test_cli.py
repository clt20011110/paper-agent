import json

from paper_agent.cli import main


def test_doctor_emits_structured_runtime(capsys) -> None:
    assert main(["doctor"]) in {0, 1}
    payload = json.loads(capsys.readouterr().out)

    assert payload["command"] == "doctor"
    assert isinstance(payload["checks"], list)
    assert {"ready", "production_ready"} <= payload.keys()
