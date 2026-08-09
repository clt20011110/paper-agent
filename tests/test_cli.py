import json

from paper_agent.cli import main


def test_doctor_emits_structured_runtime(capsys) -> None:
    assert main(["doctor"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["python_supported"] is True
    assert payload["schema_count"] == 17
    assert payload["codex_cli"]
