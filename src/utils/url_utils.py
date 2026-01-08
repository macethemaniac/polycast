import re
from typing import Optional

def extract_polymarket_slug(text: str) -> Optional[str]:
    """
    Extract a Polymarket slug from a URL or text.
    Handles formats like:
    - https://polymarket.com/event/slug
    - polymarket.com/event/slug
    - https://polymarket.com/market/slug
    """
    if not text:
        return None
        
    # Pattern for Polymarket URLs
    # Matches /event/slug or /market/slug (legacy)
    pattern = r"polymarket\.com/(?:event|market)/([a-zA-Z0-9\-]+)"
    match = re.search(pattern, text)
    
    if match:
        return match.group(1)
        
    return None

def is_polymarket_url(text: str) -> bool:
    """Check if the text contain a Polymarket market URL."""
    return extract_polymarket_slug(text) is not None
