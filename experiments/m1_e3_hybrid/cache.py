"""
M1-E3: Cache - Simple in-memory caching for template results

Caches rendered outputs to avoid re-rendering identical facts.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import hashlib


@dataclass
class CacheEntry:
    """Single cache entry."""

    key: str
    fact_hash: str
    template_output: str
    best_template_id: str
    quality_score: float
    hit_count: int = 0


class TemplateCache:
    """Simple cache for template rendering results."""

    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: Maximum cache size before clearing old entries
        """
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash_fact(fact: Dict[str, Any]) -> str:
        """Create hash of fact for cache key."""
        fact_str = str(sorted(fact.items()))
        return hashlib.md5(fact_str.encode()).hexdigest()

    def get(self, fact: Dict[str, Any]) -> Optional[str]:
        """
        Get cached result for fact.

        Args:
            fact: Temporal fact

        Returns:
            Cached template output, or None if not cached
        """
        fact_hash = self._hash_fact(fact)

        if fact_hash in self.cache:
            entry = self.cache[fact_hash]
            entry.hit_count += 1
            self.hits += 1
            return entry.template_output

        self.misses += 1
        return None

    def set(
        self,
        fact: Dict[str, Any],
        template_output: str,
        best_template_id: str,
        quality_score: float,
    ) -> None:
        """
        Cache a result.

        Args:
            fact: Temporal fact
            template_output: Rendered template output
            best_template_id: ID of template used
            quality_score: Quality of output 0-1
        """
        if len(self.cache) >= self.max_size:
            self._evict_least_used()

        fact_hash = self._hash_fact(fact)

        self.cache[fact_hash] = CacheEntry(
            key=fact_hash,
            fact_hash=fact_hash,
            template_output=template_output,
            best_template_id=best_template_id,
            quality_score=quality_score,
        )

    def _evict_least_used(self) -> None:
        """Remove least-used cache entry."""
        if not self.cache:
            return

        # Find entry with lowest hit count
        min_entry = min(self.cache.values(), key=lambda e: e.hit_count)
        del self.cache[min_entry.key]

    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total,
        }


if __name__ == "__main__":
    cache = TemplateCache()

    test_fact = {"subject": "Test", "date": "2023-01-01"}

    # Miss
    result = cache.get(test_fact)
    print(f"Cache miss: {result}")

    # Set
    cache.set(test_fact, "Test output", "test_template_1", 0.85)

    # Hit
    result = cache.get(test_fact)
    print(f"Cache hit: {result}")

    # Stats
    stats = cache.get_stats()
    print(f"Stats: {stats}")
