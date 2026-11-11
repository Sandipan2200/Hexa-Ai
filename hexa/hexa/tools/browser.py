"""Browser tab opening and search tools."""
from __future__ import annotations

import logging
import urllib.parse
import webbrowser
from typing import Annotated, Literal

from hexa.tools import tool
from hexa.config import get_config

logger = logging.getLogger(__name__)


def _get_site_whitelist() -> dict[str, str]:
    """Get the current site whitelist from config."""
    config = get_config()
    return config.sites


def _get_site_keys() -> list[str]:
    """Get the list of valid site keys."""
    return list(_get_site_whitelist().keys())


@tool
def open_tabs(sites: Annotated[list[str], "List of site keys from whitelist"]) -> str:
    """Open multiple sites as Chrome tabs from the whitelist.

    Args:
        sites: List of site keys (must be from configured whitelist: youtube, gmail, github, etc.)
    """
    site_map = _get_site_whitelist()
    valid_keys = set(site_map.keys())

    # Validate all sites against whitelist
    invalid = [s for s in sites if s.lower().strip() not in valid_keys]
    if invalid:
        available = ", ".join(sorted(valid_keys))
        logger.warning("Unknown site(s) requested: %s", invalid)
        return f"Unknown site(s): {', '.join(invalid)}. Available: {available}"

    valid_sites = [site_map[s.lower().strip()] for s in sites]

    for url in valid_sites:
        try:
            logger.info("Opening tab: %s", url)
            webbrowser.open_new_tab(url)
        except Exception as e:
            logger.exception("Failed to open tab: %s", url)

    return f"Opened {len(valid_sites)} tab(s): {', '.join(sites)}"


@tool
def search(site: str, query: str) -> str:
    """Search on a supported site using URL templates."""
    config = get_config()
    templates = config.search_templates

    site_lower = site.lower().strip()
    if site_lower not in templates:
        available = ", ".join(sorted(templates.keys()))
        logger.warning("Unknown search site: %s (available: %s)", site, available)
        return f"Unknown search site: {site}. Available: {available}"

    encoded_query = urllib.parse.quote(query)
    url = templates[site_lower].format(query=encoded_query)

    try:
        logger.info("Searching %s for: %s", site, query)
        webbrowser.open_new_tab(url)
        return f"Searching {site} for '{query}'"
    except Exception as e:
        logger.exception("Failed to search: %s", site)
        return f"Failed to search {site}: {e}"