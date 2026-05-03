from knowledge_hub.web.crawler import WebCrawler
from knowledge_hub.web.ingest import WebIngestService, make_web_note_id
from knowledge_hub.web.crawl4ai_adapter import CrawlDocument, is_crawl4ai_available
from knowledge_hub.web.ontology_extractor import WebOntologyExtractor
from knowledge_hub.web.ontology_graph import OntologyGraphService, OntologyGraphResult

__all__ = [
    "WebCrawler",
    "WebIngestService",
    "make_web_note_id",
    "CrawlDocument",
    "is_crawl4ai_available",
    "WebOntologyExtractor",
    "OntologyGraphService",
    "OntologyGraphResult",
]
