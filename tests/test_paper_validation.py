"""paper_cmd.py arxiv_id 검증 + 에러 핸들링 테스트"""

from __future__ import annotations

import pytest
import click

from knowledge_hub.cli.paper_cmd import _validate_arxiv_id


class TestArxivIdValidation:
    def test_valid_ids(self):
        assert _validate_arxiv_id("2501.06322") == "2501.06322"
        assert _validate_arxiv_id("1409.3215") == "1409.3215"
        assert _validate_arxiv_id("2501.06322v2") == "2501.06322v2"

    def test_strips_whitespace(self):
        assert _validate_arxiv_id("  2501.06322  ") == "2501.06322"

    def test_rejects_garbage(self):
        with pytest.raises(click.BadParameter, match="유효하지 않은"):
            _validate_arxiv_id("not-an-id")

    def test_rejects_empty(self):
        with pytest.raises(click.BadParameter):
            _validate_arxiv_id("")

    def test_rejects_url(self):
        with pytest.raises(click.BadParameter):
            _validate_arxiv_id("https://arxiv.org/abs/2501.06322")

    def test_rejects_partial(self):
        with pytest.raises(click.BadParameter):
            _validate_arxiv_id("2501")
