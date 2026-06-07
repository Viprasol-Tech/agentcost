from __future__ import annotations

from typer.testing import CliRunner

from agentcost.cli import app, fake_llm
from agentcost.recorder import CostStore

runner = CliRunner()


def test_fake_llm_deterministic():
    text1, usage1 = fake_llm("hello", reply="world")
    text2, usage2 = fake_llm("hello", reply="world")
    assert text1 == text2 == "world"
    assert usage1 == usage2
    assert usage1["input_tokens"] > 0


def test_demo_runs():
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "Cost by feature" in result.stdout
    assert "Cost by user" in result.stdout
    assert "Cost by agent_run" in result.stdout
    assert "Cost by model" in result.stdout
    assert "Top spenders" in result.stdout


def test_demo_save_and_report(tmp_path):
    path = tmp_path / "store.json"
    result = runner.invoke(app, ["demo", "--save", str(path)])
    assert result.exit_code == 0
    assert path.exists()
    store = CostStore.load(path)
    assert len(store) > 0

    report_result = runner.invoke(app, ["report", str(path), "--by", "user"])
    assert report_result.exit_code == 0
    assert "Cost by user" in report_result.stdout


def test_report_missing_file():
    result = runner.invoke(app, ["report", "no-such-file.json"])
    assert result.exit_code == 1


def test_models_command():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "gpt-4o" in result.stdout
