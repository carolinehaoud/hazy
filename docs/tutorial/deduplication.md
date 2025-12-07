# Tutorial: Stream Deduplication

Learn how to detect and filter duplicate items in real-time data streams using Bloom filters and related structures.

## The Problem

You're processing a stream of events (URLs, transaction IDs, log entries) and need to:

1. **Detect duplicates** in real-time
2. **Use bounded memory** regardless of stream size
3. **Handle high throughput** with low latency
4. **Accept occasional false positives** (marking new items as duplicates)

## Use Cases

- **Web crawler**: Don't re-crawl URLs you've already visited
- **Event processing**: Ensure exactly-once semantics
- **Log deduplication**: Filter repeated log entries
- **Ad deduplication**: Don't show the same ad twice

## Solution: Bloom Filter Deduplication

```python
from hazy import BloomFilter, ScalableBloomFilter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Callable
import hashlib
import time


@dataclass
class Event:
    """A generic event with an ID and payload."""
    id: str
    payload: dict
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class StreamDeduplicator:
    """
    Deduplicate streaming events using a Bloom filter.

    Events that have been seen before are filtered out.
    False positives (new events marked as duplicates) are possible
    but false negatives (duplicates marked as new) are not.
    """

    def __init__(
        self,
        expected_items: int = 1_000_000,
        false_positive_rate: float = 0.01,
        scalable: bool = False
    ):
        """
        Initialize the deduplicator.

        Args:
            expected_items: Expected number of unique items
            false_positive_rate: Acceptable false positive rate
            scalable: Use ScalableBloomFilter for unknown cardinality
        """
        if scalable:
            self.seen = ScalableBloomFilter(
                initial_capacity=expected_items // 10,
                false_positive_rate=false_positive_rate
            )
        else:
            self.seen = BloomFilter(
                expected_items=expected_items,
                false_positive_rate=false_positive_rate
            )

        self.stats = {
            "total_processed": 0,
            "unique": 0,
            "duplicates": 0,
        }

    def is_duplicate(self, event_id: str) -> bool:
        """Check if an event ID has been seen before."""
        return event_id in self.seen

    def mark_seen(self, event_id: str):
        """Mark an event ID as seen."""
        self.seen.add(event_id)

    def process(self, event: Event) -> bool:
        """
        Process an event, returning True if it's new (not a duplicate).

        Args:
            event: The event to process

        Returns:
            True if the event is new, False if it's a duplicate
        """
        self.stats["total_processed"] += 1

        if self.is_duplicate(event.id):
            self.stats["duplicates"] += 1
            return False

        self.mark_seen(event.id)
        self.stats["unique"] += 1
        return True

    def process_stream(
        self,
        events: Iterator[Event],
        handler: Callable[[Event], None]
    ) -> dict:
        """
        Process a stream of events, calling handler for each unique event.

        Args:
            events: Iterator of events
            handler: Function to call for each unique event

        Returns:
            Processing statistics
        """
        for event in events:
            if self.process(event):
                handler(event)

        return self.stats

    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            **self.stats,
            "duplicate_rate": (
                self.stats["duplicates"] / max(1, self.stats["total_processed"])
            ),
            "filter_fill_ratio": getattr(self.seen, "fill_ratio", None),
            "memory_bytes": self.seen.size_in_bytes,
        }
```

## Example: URL Deduplication for Web Crawler

```python
import random
from urllib.parse import urlparse


def generate_urls(n_urls: int, duplicate_rate: float = 0.3) -> Iterator[Event]:
    """Generate a stream of URLs with some duplicates."""

    domains = [
        "example.com", "test.org", "sample.net",
        "demo.io", "website.com"
    ]
    paths = [
        "/", "/about", "/products", "/blog", "/contact",
        "/page1", "/page2", "/page3", "/article/1", "/article/2"
    ]

    seen_urls = []

    for i in range(n_urls):
        # Sometimes generate a duplicate
        if seen_urls and random.random() < duplicate_rate:
            url = random.choice(seen_urls)
        else:
            domain = random.choice(domains)
            path = random.choice(paths)
            query = f"?id={random.randint(1, 1000)}" if random.random() > 0.5 else ""
            url = f"https://{domain}{path}{query}"
            seen_urls.append(url)

            # Limit memory for seen URLs
            if len(seen_urls) > 10000:
                seen_urls = seen_urls[-5000:]

        yield Event(id=url, payload={"url": url})


# Create deduplicator
dedup = StreamDeduplicator(
    expected_items=100_000,
    false_positive_rate=0.001  # 0.1% FPR for crawlers
)

# Track unique URLs we'd crawl
unique_urls = []

def mock_crawl(event: Event):
    """Simulate crawling a URL."""
    unique_urls.append(event.payload["url"])


# Process stream
print("Processing URL stream...")
start = time.time()
stats = dedup.process_stream(
    generate_urls(n_urls=500_000, duplicate_rate=0.4),
    handler=mock_crawl
)
elapsed = time.time() - start

# Report results
print(f"\n{'='*50}")
print("URL DEDUPLICATION RESULTS")
print(f"{'='*50}")
print(f"Total URLs processed:  {stats['total_processed']:,}")
print(f"Unique URLs (crawled): {stats['unique']:,}")
print(f"Duplicates filtered:   {stats['duplicates']:,}")
print(f"Duplicate rate:        {stats['duplicate_rate']:.1%}")
print(f"Processing time:       {elapsed:.2f}s")
print(f"Throughput:            {stats['total_processed']/elapsed:,.0f} URLs/sec")
print(f"Memory used:           {dedup.get_stats()['memory_bytes'] / 1024:.1f} KB")
```

## Advanced: Time-Windowed Deduplication

For streams where you only care about recent duplicates:

```python
from hazy import BloomFilter
from collections import deque
from datetime import datetime, timedelta


class TimeWindowedDeduplicator:
    """
    Deduplicate events within a time window.

    Uses rotating Bloom filters to "forget" old events.
    """

    def __init__(
        self,
        window_minutes: int = 60,
        buckets: int = 6,
        expected_items_per_bucket: int = 100_000,
        false_positive_rate: float = 0.01
    ):
        """
        Initialize time-windowed deduplicator.

        Args:
            window_minutes: Total time window for deduplication
            buckets: Number of time buckets (more = smoother rotation)
            expected_items_per_bucket: Expected items per time bucket
            false_positive_rate: FPR for each bucket
        """
        self.window_minutes = window_minutes
        self.bucket_minutes = window_minutes // buckets
        self.expected_items = expected_items_per_bucket
        self.fpr = false_positive_rate

        # Create initial buckets
        self.buckets = deque(maxlen=buckets)
        self.bucket_timestamps = deque(maxlen=buckets)
        self._create_bucket()

        self.stats = {"processed": 0, "unique": 0, "duplicates": 0}

    def _create_bucket(self):
        """Create a new time bucket."""
        self.buckets.append(BloomFilter(
            expected_items=self.expected_items,
            false_positive_rate=self.fpr
        ))
        self.bucket_timestamps.append(datetime.now())

    def _maybe_rotate(self):
        """Create new bucket if current one is old enough."""
        if not self.bucket_timestamps:
            self._create_bucket()
            return

        age = datetime.now() - self.bucket_timestamps[-1]
        if age > timedelta(minutes=self.bucket_minutes):
            self._create_bucket()

    def is_duplicate(self, event_id: str) -> bool:
        """Check if event was seen in any recent bucket."""
        for bucket in self.buckets:
            if event_id in bucket:
                return True
        return False

    def process(self, event_id: str) -> bool:
        """
        Process an event, returning True if new within the time window.
        """
        self._maybe_rotate()
        self.stats["processed"] += 1

        if self.is_duplicate(event_id):
            self.stats["duplicates"] += 1
            return False

        # Add to most recent bucket
        self.buckets[-1].add(event_id)
        self.stats["unique"] += 1
        return True

    def get_stats(self) -> dict:
        """Get statistics."""
        total_memory = sum(b.size_in_bytes for b in self.buckets)
        return {
            **self.stats,
            "active_buckets": len(self.buckets),
            "memory_bytes": total_memory,
            "window_minutes": self.window_minutes,
        }


# Example usage
print("\n" + "="*50)
print("TIME-WINDOWED DEDUPLICATION")
print("="*50)

windowed_dedup = TimeWindowedDeduplicator(
    window_minutes=60,  # 1 hour window
    buckets=6,          # 10-minute buckets
    expected_items_per_bucket=10_000
)

# Simulate events
for i in range(50_000):
    event_id = f"event_{i % 5000}"  # Creates duplicates
    is_new = windowed_dedup.process(event_id)

stats = windowed_dedup.get_stats()
print(f"Processed:      {stats['processed']:,}")
print(f"Unique:         {stats['unique']:,}")
print(f"Duplicates:     {stats['duplicates']:,}")
print(f"Active buckets: {stats['active_buckets']}")
print(f"Memory:         {stats['memory_bytes'] / 1024:.1f} KB")
```

## Deduplication with Deletion: Cuckoo Filter

When you need to **remove** items (e.g., allow re-processing after some time):

```python
from hazy import CuckooFilter


class DeletableDeduplicator:
    """
    Deduplicator that supports removing items.

    Uses a Cuckoo filter instead of Bloom filter.
    """

    def __init__(self, capacity: int = 1_000_000):
        self.seen = CuckooFilter(capacity=capacity)
        self.stats = {"processed": 0, "unique": 0, "duplicates": 0, "removed": 0}

    def process(self, event_id: str) -> bool:
        """Process an event, returning True if new."""
        self.stats["processed"] += 1

        if event_id in self.seen:
            self.stats["duplicates"] += 1
            return False

        if not self.seen.add(event_id):
            # Filter is full - this shouldn't happen if sized correctly
            raise RuntimeError("Cuckoo filter is full")

        self.stats["unique"] += 1
        return True

    def remove(self, event_id: str) -> bool:
        """
        Remove an event, allowing it to be processed again.

        Returns True if the event was present and removed.
        """
        if event_id in self.seen:
            self.seen.remove(event_id)
            self.stats["removed"] += 1
            return True
        return False


# Example: Process events then allow re-processing
deletable_dedup = DeletableDeduplicator(capacity=10_000)

# Process some events
for i in range(1000):
    deletable_dedup.process(f"event_{i}")

print(f"\nAfter processing: {deletable_dedup.stats['unique']} unique events")

# Remove some events to allow re-processing
for i in range(100):
    deletable_dedup.remove(f"event_{i}")

print(f"After removal: {deletable_dedup.stats['removed']} events can be re-processed")

# Re-process removed events
reprocessed = 0
for i in range(100):
    if deletable_dedup.process(f"event_{i}"):
        reprocessed += 1

print(f"Re-processed: {reprocessed} events")
```

## Performance Comparison

```python
import time
from hazy import BloomFilter, CuckooFilter, ScalableBloomFilter


def benchmark_deduplication(n_items: int = 1_000_000, duplicate_rate: float = 0.3):
    """Benchmark different deduplication approaches."""

    # Generate test data
    items = []
    for i in range(n_items):
        if items and random.random() < duplicate_rate:
            items.append(random.choice(items[:len(items)//2]))
        else:
            items.append(f"item_{i}")

    results = {}

    # 1. Python set (baseline - exact)
    start = time.time()
    seen_set = set()
    unique_set = 0
    for item in items:
        if item not in seen_set:
            seen_set.add(item)
            unique_set += 1
    results["Python set"] = {
        "time": time.time() - start,
        "memory": len(seen_set) * 50,  # Approximate
        "unique": unique_set,
    }

    # 2. Bloom Filter
    bf = BloomFilter(expected_items=n_items, false_positive_rate=0.01)
    start = time.time()
    unique_bf = 0
    for item in items:
        if item not in bf:
            bf.add(item)
            unique_bf += 1
    results["BloomFilter"] = {
        "time": time.time() - start,
        "memory": bf.size_in_bytes,
        "unique": unique_bf,
    }

    # 3. Cuckoo Filter
    cf = CuckooFilter(capacity=n_items)
    start = time.time()
    unique_cf = 0
    for item in items:
        if item not in cf:
            cf.add(item)
            unique_cf += 1
    results["CuckooFilter"] = {
        "time": time.time() - start,
        "memory": cf.size_in_bytes,
        "unique": unique_cf,
    }

    # 4. Scalable Bloom Filter
    sbf = ScalableBloomFilter(initial_capacity=n_items // 10)
    start = time.time()
    unique_sbf = 0
    for item in items:
        if item not in sbf:
            sbf.add(item)
            unique_sbf += 1
    results["ScalableBloomFilter"] = {
        "time": time.time() - start,
        "memory": sbf.size_in_bytes,
        "unique": unique_sbf,
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"DEDUPLICATION BENCHMARK ({n_items:,} items, {duplicate_rate:.0%} duplicates)")
    print(f"{'='*60}")
    print(f"{'Method':<22} {'Time':>10} {'Memory':>12} {'Unique':>10}")
    print("-" * 60)

    for method, data in results.items():
        print(f"{method:<22} {data['time']:>9.2f}s {data['memory']/1024:>10.1f}KB {data['unique']:>10,}")


benchmark_deduplication(n_items=1_000_000, duplicate_rate=0.3)
```

## Best Practices

### 1. Choose the Right Structure

| Need | Use |
|------|-----|
| Simple deduplication | `BloomFilter` |
| Unknown stream size | `ScalableBloomFilter` |
| Need to remove items | `CuckooFilter` |
| Time-windowed | Rotating `BloomFilter`s |

### 2. Size Appropriately

```python
# Rule of thumb: size for 2x expected items
bf = BloomFilter(
    expected_items=expected_unique * 2,
    false_positive_rate=0.01
)
```

### 3. Monitor Fill Ratio

```python
if bf.fill_ratio > 0.5:
    print("Warning: Filter getting full, FPR increasing")
```

### 4. Consider Persistence

```python
# Save filter state for recovery
dedup.seen.save("dedup_state.hazy")

# Restore on restart
dedup.seen = BloomFilter.load("dedup_state.hazy")
```

## Next Tutorial

Continue to [Similarity Search](similarity-search.md) to learn how to find similar documents using MinHash.
