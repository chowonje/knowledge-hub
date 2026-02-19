"""main.py 공통 에러 핸들러 테스트"""

from __future__ import annotations

from click.testing import CliRunner

from knowledge_hub.cli.main import cli


class TestCLIErrorHandler:
    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "knowledge-hub" in result.output

    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Knowledge Hub" in result.output

    def test_unknown_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["nonexistent-command"])
        assert result.exit_code != 0

    def test_verbose_flag_accepted(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0
