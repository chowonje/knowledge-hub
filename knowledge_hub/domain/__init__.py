"""Domain-layer root.

This package keeps shared domain models plus source-specific domain packs.
"""

from knowledge_hub.domain.models import *  # noqa: F403
from knowledge_hub.domain.protocols import DomainPack, PaperDomainPack
from knowledge_hub.domain.registry import (
    DomainRegistration,
    get_domain_pack,
    get_domain_registration,
    list_domain_registrations,
    normalize_domain_source,
    resolve_domain_name,
)
