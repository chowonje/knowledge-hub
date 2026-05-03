"""Canonical CLI command modules."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "ask": "knowledge_hub.interfaces.cli.commands.search_cmd",
    "agent_group": "knowledge_hub.interfaces.cli.commands.agent_cmd",
    "belief_group": "knowledge_hub.interfaces.cli.commands.belief_cmd",
    "claims_group": "knowledge_hub.interfaces.cli.commands.claims_cmd",
    "config_group": "knowledge_hub.interfaces.cli.commands.config_cmd",
    "crawl_group": "knowledge_hub.interfaces.cli.commands.crawl_cmd",
    "decision_group": "knowledge_hub.interfaces.cli.commands.decision_cmd",
    "discover": "knowledge_hub.interfaces.cli.commands.discover_cmd",
    "explore_group": "knowledge_hub.interfaces.cli.commands.explore_cmd",
    "feature_group": "knowledge_hub.interfaces.cli.commands.feature_cmd",
    "graph_group": "knowledge_hub.interfaces.cli.commands.graph_cmd",
    "health_cmd": "knowledge_hub.interfaces.cli.commands.health_cmd",
    "index_cmd": "knowledge_hub.interfaces.cli.commands.index_cmd",
    "init_cmd": "knowledge_hub.interfaces.cli.commands.init_cmd",
    "learn_group": "knowledge_hub.interfaces.cli.commands.learn_cmd",
    "mcp_cmd": "knowledge_hub.interfaces.cli.commands.mcp_cmd",
    "ontology_group": "knowledge_hub.interfaces.cli.commands.ontology_cmd",
    "ops_action_ack": "knowledge_hub.interfaces.cli.commands.ops_cmd",
    "ops_action_execute": "knowledge_hub.interfaces.cli.commands.ops_cmd",
    "ops_action_list": "knowledge_hub.interfaces.cli.commands.ops_cmd",
    "ops_action_receipts": "knowledge_hub.interfaces.cli.commands.ops_cmd",
    "ops_action_resolve": "knowledge_hub.interfaces.cli.commands.ops_cmd",
    "ops_report_run": "knowledge_hub.interfaces.cli.commands.ops_cmd",
    "outcome_group": "knowledge_hub.interfaces.cli.commands.outcome_cmd",
    "paper_group": "knowledge_hub.interfaces.cli.commands.paper_cmd",
    "run_status": "knowledge_hub.interfaces.cli.commands.status_cmd",
    "search": "knowledge_hub.interfaces.cli.commands.search_cmd",
    "setup_cmd": "knowledge_hub.interfaces.cli.commands.setup_cmd",
    "vault_group": "knowledge_hub.interfaces.cli.commands.vault_cmd",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)
