"""LLM-powered semantic market analysis.

Uses Claude/GPT to extract structured market features for better clustering
and relationship discovery, based on arXiv paper 2512.02436.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# Cache directory
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SEMANTIC_CACHE_FILE = DATA_DIR / "semantic_analysis.json"


@dataclass
class MarketAnalysis:
    """Structured analysis of a prediction market."""
    market_id: str
    platform: str

    # Core extraction
    core_event: str = ""
    entities: List[str] = field(default_factory=list)
    time_constraints: List[str] = field(default_factory=list)
    outcome_conditions: str = ""

    # Classification
    category: str = ""  # politics, crypto, sports, economy, tech, world, entertainment
    subcategory: str = ""
    topics: List[str] = field(default_factory=list)

    # Dependencies
    related_topics: List[str] = field(default_factory=list)
    logical_dependencies: List[str] = field(default_factory=list)

    # Metadata
    question_type: str = ""  # binary, numeric_threshold, date_based, conditional
    specificity_score: float = 0.0  # 0-1, how specific/narrow is the question
    analyzed_at: float = field(default_factory=time.time)
    content_hash: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "MarketAnalysis":
        return cls(**d)


class SemanticAnalysisCache:
    """Cache for semantic analysis results."""

    def __init__(self):
        self.cache: Dict[str, MarketAnalysis] = {}
        self._load()

    def _load(self) -> None:
        if SEMANTIC_CACHE_FILE.exists():
            try:
                with open(SEMANTIC_CACHE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, val in data.items():
                        self.cache[key] = MarketAnalysis.from_dict(val)
                logger.info(f"Loaded {len(self.cache)} cached semantic analyses")
            except Exception as e:
                logger.debug(f"Error loading semantic cache: {e}")

    def _save(self) -> None:
        try:
            data = {k: v.to_dict() for k, v in self.cache.items()}
            with open(SEMANTIC_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Error saving semantic cache: {e}")

    def get(self, market_id: str, content_hash: str) -> Optional[MarketAnalysis]:
        """Get cached analysis if content hasn't changed."""
        cached = self.cache.get(market_id)
        if cached and cached.content_hash == content_hash:
            return cached
        return None

    def set(self, analysis: MarketAnalysis) -> None:
        self.cache[analysis.market_id] = analysis
        self._save()


class SemanticAnalyzer:
    """LLM-powered semantic market analyzer."""

    # Category keywords for fast classification
    CATEGORY_KEYWORDS = {
        "politics": ["election", "president", "senate", "congress", "vote", "poll", "trump", "biden", "democrat", "republican", "governor", "mayor", "political"],
        "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "token", "defi", "nft", "altcoin", "binance", "coinbase"],
        "sports": ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "hockey", "championship", "playoff", "super bowl", "world cup"],
        "economy": ["gdp", "inflation", "fed", "interest rate", "unemployment", "stock", "s&p", "nasdaq", "dow", "recession", "economic"],
        "tech": ["ai", "openai", "google", "apple", "microsoft", "meta", "tesla", "spacex", "launch", "release", "iphone"],
        "world": ["war", "conflict", "nato", "china", "russia", "ukraine", "israel", "gaza", "international", "treaty"],
        "entertainment": ["movie", "oscar", "grammy", "album", "box office", "streaming", "netflix", "disney"],
    }

    # Entity patterns
    ENTITY_PATTERNS = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names like "Donald Trump"
        r'\b[A-Z]{2,}\b',  # Acronyms like "GDP", "BTC"
        r'\$[A-Z]+\b',  # Tickers like "$BTC"
        r'\b\d{4}\b',  # Years
    ]

    def __init__(self, use_llm: bool = False):
        """
        Initialize analyzer.

        Args:
            use_llm: If True, use Claude API for deep analysis (costs money).
                    If False, use fast heuristic analysis (free).
        """
        self.use_llm = use_llm
        self.cache = SemanticAnalysisCache()
        self._anthropic_client = None

    def _get_content_hash(self, question: str, description: str = "") -> str:
        """Generate hash for content-based caching."""
        content = f"{question}:{description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_entities_fast(self, text: str) -> List[str]:
        """Fast entity extraction using patterns."""
        entities = set()

        for pattern in self.ENTITY_PATTERNS:
            matches = re.findall(pattern, text)
            entities.update(matches)

        # Remove common false positives
        false_positives = {"YES", "NO", "THE", "AND", "FOR", "BUT", "NOT"}
        entities = [e for e in entities if e not in false_positives]

        return list(entities)[:10]  # Limit to 10 entities

    def _classify_category_fast(self, text: str) -> Tuple[str, str]:
        """Fast category classification using keywords."""
        text_lower = text.lower()

        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return "other", ""

        category = max(scores, key=scores.get)

        # Subcategory detection
        subcategory = ""
        if category == "politics":
            if "election" in text_lower or "vote" in text_lower:
                subcategory = "elections"
            elif "congress" in text_lower or "senate" in text_lower:
                subcategory = "legislation"
        elif category == "crypto":
            if "bitcoin" in text_lower or "btc" in text_lower:
                subcategory = "bitcoin"
            elif "ethereum" in text_lower or "eth" in text_lower:
                subcategory = "ethereum"
        elif category == "sports":
            for sport in ["nfl", "nba", "mlb", "nhl", "soccer"]:
                if sport in text_lower:
                    subcategory = sport
                    break

        return category, subcategory

    def _extract_time_constraints(self, text: str) -> List[str]:
        """Extract time-related constraints from text."""
        constraints = []

        # Date patterns
        date_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\bby (end of|the end of)?\s*\d{4}\b',
            r'\bbefore\s+\w+\s+\d{4}\b',
            r'\bin\s+Q[1-4]\s+\d{4}\b',
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            constraints.extend(matches)

        return constraints[:5]

    def _detect_question_type(self, question: str) -> str:
        """Detect the type of prediction question."""
        q_lower = question.lower()

        if any(word in q_lower for word in ["more than", "less than", "above", "below", "at least", "exceed"]):
            return "numeric_threshold"
        elif any(word in q_lower for word in ["by", "before", "on", "during"]):
            return "date_based"
        elif "if" in q_lower or "given" in q_lower:
            return "conditional"
        else:
            return "binary"

    def _calculate_specificity(self, analysis: MarketAnalysis) -> float:
        """Calculate how specific/narrow a market question is."""
        score = 0.0

        # More entities = more specific
        score += min(len(analysis.entities) * 0.1, 0.3)

        # Time constraints = more specific
        score += min(len(analysis.time_constraints) * 0.15, 0.3)

        # Longer outcome conditions = more specific
        if len(analysis.outcome_conditions) > 50:
            score += 0.2
        elif len(analysis.outcome_conditions) > 20:
            score += 0.1

        # Subcategory = more specific
        if analysis.subcategory:
            score += 0.2

        return min(score, 1.0)

    def analyze_market_fast(self, market: Dict) -> MarketAnalysis:
        """
        Fast heuristic analysis without LLM calls.
        Good for batch processing and when LLM costs matter.
        """
        market_id = market.get("id") or market.get("market_id") or market.get("ticker") or ""
        platform = market.get("platform") or "unknown"
        question = market.get("question") or market.get("title") or ""
        description = market.get("description") or market.get("rules") or ""

        content_hash = self._get_content_hash(question, description)

        # Check cache
        cached = self.cache.get(market_id, content_hash)
        if cached:
            return cached

        # Extract features
        full_text = f"{question} {description}"

        entities = self._extract_entities_fast(full_text)
        category, subcategory = self._classify_category_fast(full_text)
        time_constraints = self._extract_time_constraints(full_text)
        question_type = self._detect_question_type(question)

        # Build analysis
        analysis = MarketAnalysis(
            market_id=market_id,
            platform=platform,
            core_event=question[:200],  # Truncated question as core event
            entities=entities,
            time_constraints=time_constraints,
            outcome_conditions=description[:500] if description else question,
            category=category,
            subcategory=subcategory,
            topics=[category, subcategory] if subcategory else [category],
            related_topics=list(set(entities) - set([category, subcategory])),
            logical_dependencies=[],
            question_type=question_type,
            content_hash=content_hash,
        )

        analysis.specificity_score = self._calculate_specificity(analysis)

        # Cache result
        self.cache.set(analysis)

        return analysis

    def analyze_market_llm(self, market: Dict) -> MarketAnalysis:
        """
        Deep analysis using Claude API.
        More accurate but costs money. Use sparingly.
        """
        # First try fast analysis as baseline
        fast_analysis = self.analyze_market_fast(market)

        if not self.use_llm:
            return fast_analysis

        # Check if we have API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.debug("No ANTHROPIC_API_KEY, falling back to fast analysis")
            return fast_analysis

        try:
            if self._anthropic_client is None:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=api_key)

            question = market.get("question") or market.get("title") or ""
            description = market.get("description") or market.get("rules") or ""

            prompt = f"""Analyze this prediction market and extract structured information.

Question: {question}
Description: {description[:1000]}

Return a JSON object with these fields:
- core_event: The main event being predicted (1 sentence)
- entities: List of key entities (people, orgs, assets) mentioned
- time_constraints: Any deadlines or time limits
- category: One of [politics, crypto, sports, economy, tech, world, entertainment, other]
- subcategory: More specific category
- topics: List of relevant topics
- related_topics: Topics that could affect this outcome
- logical_dependencies: Other events this depends on

Only return valid JSON, no explanation."""

            response = self._anthropic_client.messages.create(
                model="claude-3-haiku-20240307",  # Fast and cheap
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            result_text = response.content[0].text
            result = json.loads(result_text)

            # Update fast analysis with LLM results
            fast_analysis.core_event = result.get("core_event", fast_analysis.core_event)
            fast_analysis.entities = result.get("entities", fast_analysis.entities)
            fast_analysis.category = result.get("category", fast_analysis.category)
            fast_analysis.subcategory = result.get("subcategory", fast_analysis.subcategory)
            fast_analysis.topics = result.get("topics", fast_analysis.topics)
            fast_analysis.related_topics = result.get("related_topics", fast_analysis.related_topics)
            fast_analysis.logical_dependencies = result.get("logical_dependencies", fast_analysis.logical_dependencies)

            # Update specificity with new info
            fast_analysis.specificity_score = self._calculate_specificity(fast_analysis)

            # Cache updated analysis
            self.cache.set(fast_analysis)

            return fast_analysis

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, using fast analysis")
            return fast_analysis

    def analyze_market(self, market: Dict) -> MarketAnalysis:
        """
        Analyze a market using the configured method.
        """
        if self.use_llm:
            return self.analyze_market_llm(market)
        return self.analyze_market_fast(market)

    def analyze_batch(self, markets: List[Dict]) -> List[MarketAnalysis]:
        """Analyze multiple markets efficiently."""
        results = []
        for market in markets:
            try:
                analysis = self.analyze_market(market)
                results.append(analysis)
            except Exception as e:
                logger.debug(f"Error analyzing market: {e}")
                continue
        return results

    def get_category_distribution(self, analyses: List[MarketAnalysis]) -> Dict[str, int]:
        """Get distribution of categories."""
        dist = {}
        for a in analyses:
            cat = a.category or "unknown"
            dist[cat] = dist.get(cat, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    def find_similar_topics(self, analysis: MarketAnalysis, all_analyses: List[MarketAnalysis]) -> List[Tuple[MarketAnalysis, float]]:
        """Find markets with similar topics."""
        similar = []

        my_topics = set(analysis.topics + analysis.entities)
        if not my_topics:
            return []

        for other in all_analyses:
            if other.market_id == analysis.market_id:
                continue

            other_topics = set(other.topics + other.entities)
            if not other_topics:
                continue

            # Jaccard similarity
            intersection = len(my_topics & other_topics)
            union = len(my_topics | other_topics)
            similarity = intersection / union if union > 0 else 0

            if similarity > 0.1:
                similar.append((other, similarity))

        return sorted(similar, key=lambda x: -x[1])[:20]


# Convenience functions
_analyzer: Optional[SemanticAnalyzer] = None


def get_semantic_analyzer(use_llm: bool = False) -> SemanticAnalyzer:
    """Get or create singleton analyzer."""
    global _analyzer
    if _analyzer is None or _analyzer.use_llm != use_llm:
        _analyzer = SemanticAnalyzer(use_llm=use_llm)
    return _analyzer


def analyze_market(market: Dict, use_llm: bool = False) -> MarketAnalysis:
    """Analyze a single market."""
    analyzer = get_semantic_analyzer(use_llm)
    return analyzer.analyze_market(market)


def analyze_markets_batch(markets: List[Dict], use_llm: bool = False) -> List[MarketAnalysis]:
    """Analyze multiple markets."""
    analyzer = get_semantic_analyzer(use_llm)
    return analyzer.analyze_batch(markets)
