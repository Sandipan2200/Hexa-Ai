"""Browser tab opening and search tools."""
from __future__ import annotations

import logging
import urllib.parse
import webbrowser
from hexa.tools import tool
from hexa.config import get_config

logger = logging.getLogger(__name__)


@tool
def open_tabs(sites: list[str]) -> str:
    """Open multiple sites as Chrome tabs from the whitelist."""
    config = get_config()
    site_map = config.sites

    valid_sites = []
    for site in sites:
        site_lower = site.lower().strip()
        if site_lower in site_map:
            valid_sites.append(site_map[site_lower])
        else:
            logger.warning("Unknown site requested: %s", site)

    if not valid_sites:
        available = ", ".join(sorted(site_map.keys()))
        return f"No valid sites. Available: {available}"

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