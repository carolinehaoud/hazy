# Tutorial: Web Analytics Dashboard

In this tutorial, we'll build a memory-efficient web analytics system that tracks unique visitors, page views, and trending content using probabilistic data structures.

## The Problem

You're building analytics for a website with millions of page views per day. You need to track:

1. **Unique visitors** per day (cardinality)
2. **Page view counts** for each URL (frequency)
3. **Trending pages** (top-k)
4. **Returning vs new visitors** (set membership)

Storing every event exactly would require gigabytes of memory. Instead, we'll use probabilistic structures that give us accurate-enough answers with megabytes.

## Solution Overview

| Metric | Data Structure | Memory | Accuracy |
|--------|---------------|--------|----------|
| Unique visitors | HyperLogLog | 16 KB | ~1% error |
| Page view counts | Count-Min Sketch | 400 KB | Slight overcount |
| Trending pages | TopK | ~10 KB | Exact for heavy hitters |
| Returning visitors | BloomFilter | ~1.2 MB | 1% false positive |

**Total: ~1.6 MB** to track millions of events!

## Implementation

### Step 1: Set Up the Analytics Tracker

```python
from hazy import BloomFilter, HyperLogLog, CountMinSketch, TopK
from datetime import datetime, date
from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class PageView:
    """Represents a single page view event."""
    visitor_id: str
    url: str
    timestamp: datetime
    referrer: Optional[str] = None


class WebAnalytics:
    """Memory-efficient web analytics using probabilistic data structures."""

    def __init__(
        self,
        expected_daily_visitors: int = 1_000_000,
        expected_daily_pageviews: int = 10_000_000,
        top_k: int = 100
    ):
        # Track unique visitors (HyperLogLog: ~16KB)
        self.unique_visitors = HyperLogLog(precision=14)

        # Track page view frequencies (Count-Min Sketch: ~400KB)
        self.page_views = CountMinSketch(width=10000, depth=5)

        # Track trending pages (TopK: ~10KB)
        self.trending = TopK(k=top_k)

        # Track known visitors for new vs returning (BloomFilter: ~1.2MB)
        self.known_visitors = BloomFilter(
            expected_items=expected_daily_visitors,
            false_positive_rate=0.01
        )

        # Counters
        self.total_pageviews = 0
        self.new_visitors = 0
        self.returning_visitors = 0

    def record(self, event: PageView):
        """Record a page view event."""
        self.total_pageviews += 1

        # Track unique visitors
        self.unique_visitors.add(event.visitor_id)

        # Track page popularity
        self.page_views.add(event.url)
        self.trending.add(event.url)

        # Track new vs returning
        if event.visitor_id in self.known_visitors:
            self.returning_visitors += 1
        else:
            self.new_visitors += 1
            self.known_visitors.add(event.visitor_id)

    def get_stats(self) -> dict:
        """Get current analytics statistics."""
        return {
            "total_pageviews": self.total_pageviews,
            "unique_visitors": int(self.unique_visitors.cardinality()),
            "new_visitors": self.new_visitors,
            "returning_visitors": self.returning_visitors,
            "pages_per_visitor": (
                self.total_pageviews / max(1, self.unique_visitors.cardinality())
            ),
        }

    def get_page_views(self, url: str) -> int:
        """Get estimated view count for a specific page."""
        return self.page_views[url]

    def get_trending(self, n: int = 10) -> list:
        """Get the top N trending pages."""
        return self.trending.top(n)

    def memory_usage(self) -> dict:
        """Get memory usage of each component."""
        return {
            "unique_visitors_hll": self.unique_visitors.size_in_bytes,
            "page_views_cms": self.page_views.size_in_bytes,
            "trending_topk": self.trending.size_in_bytes,
            "known_visitors_bloom": self.known_visitors.size_in_bytes,
            "total_bytes": (
                self.unique_visitors.size_in_bytes +
                self.page_views.size_in_bytes +
                self.trending.size_in_bytes +
                self.known_visitors.size_in_bytes
            ),
        }
```

### Step 2: Simulate Traffic

Let's generate realistic web traffic to test our analytics:

```python
def generate_traffic(n_events: int, n_unique_visitors: int) -> list[PageView]:
    """Generate simulated web traffic."""

    # Popular pages (Zipf distribution - few pages get most traffic)
    pages = [
        "/",
        "/about",
        "/products",
        "/products/widget-a",
        "/products/widget-b",
        "/products/widget-c",
        "/blog",
        "/blog/post-1",
        "/blog/post-2",
        "/blog/post-3",
        "/contact",
        "/pricing",
        "/signup",
        "/login",
        "/dashboard",
    ]

    # Page popularity weights (home page most popular)
    weights = [100, 20, 50, 30, 25, 20, 40, 15, 12, 10, 8, 25, 15, 30, 20]

    events = []
    visitors = [f"visitor_{i}" for i in range(n_unique_visitors)]

    # Some visitors return multiple times (power law)
    visitor_weights = [1.0 / (i + 1) ** 0.5 for i in range(n_unique_visitors)]

    for i in range(n_events):
        visitor = random.choices(visitors, weights=visitor_weights)[0]
        url = random.choices(pages, weights=weights)[0]

        events.append(PageView(
            visitor_id=visitor,
            url=url,
            timestamp=datetime.now(),
        ))

    return events


# Generate 1 million page views from 100,000 unique visitors
print("Generating simulated traffic...")
events = generate_traffic(n_events=1_000_000, n_unique_visitors=100_000)
print(f"Generated {len(events):,} events")
```

### Step 3: Process Events and Analyze

```python
# Create analytics tracker
analytics = WebAnalytics(
    expected_daily_visitors=100_000,
    expected_daily_pageviews=1_000_000
)

# Process all events
print("\nProcessing events...")
for event in events:
    analytics.record(event)

# Get statistics
stats = analytics.get_stats()
print("\n" + "=" * 50)
print("ANALYTICS SUMMARY")
print("=" * 50)
print(f"Total Page Views:    {stats['total_pageviews']:,}")
print(f"Unique Visitors:     {stats['unique_visitors']:,}")
print(f"New Visitors:        {stats['new_visitors']:,}")
print(f"Returning Visitors:  {stats['returning_visitors']:,}")
print(f"Pages per Visitor:   {stats['pages_per_visitor']:.2f}")

# Show trending pages
print("\n" + "=" * 50)
print("TOP 10 TRENDING PAGES")
print("=" * 50)
for url, count in analytics.get_trending(10):
    print(f"  {url:<30} {count:>10,} views")

# Show memory usage
memory = analytics.memory_usage()
print("\n" + "=" * 50)
print("MEMORY USAGE")
print("=" * 50)
print(f"  HyperLogLog (visitors):  {memory['unique_visitors_hll']:>10,} bytes")
print(f"  Count-Min Sketch:        {memory['page_views_cms']:>10,} bytes")
print(f"  TopK (trending):         {memory['trending_topk']:>10,} bytes")
print(f"  BloomFilter (known):     {memory['known_visitors_bloom']:>10,} bytes")
print(f"  {'─' * 40}")
print(f"  TOTAL:                   {memory['total_bytes']:>10,} bytes")
print(f"                           ({memory['total_bytes'] / 1024 / 1024:.2f} MB)")
```

### Step 4: Visualize Results

```python
import matplotlib.pyplot as plt
from hazy.viz import plot_hll, plot_cms, plot_topk


def visualize_analytics(analytics: WebAnalytics):
    """Create visualizations for the analytics data."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Web Analytics Dashboard", fontsize=16, fontweight="bold")

    # 1. HyperLogLog register distribution
    plt.sca(axes[0, 0])
    plot_hll(analytics.unique_visitors, title="Unique Visitor Counter (HLL)")

    # 2. Top pages bar chart
    plt.sca(axes[0, 1])
    plot_topk(analytics.trending, n=10, title="Top 10 Pages")

    # 3. Memory usage pie chart
    ax = axes[1, 0]
    memory = analytics.memory_usage()
    labels = ["HyperLogLog", "Count-Min Sketch", "TopK", "BloomFilter"]
    sizes = [
        memory["unique_visitors_hll"],
        memory["page_views_cms"],
        memory["trending_topk"],
        memory["known_visitors_bloom"],
    ]
    colors = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Memory Usage by Component")

    # 4. New vs Returning visitors
    ax = axes[1, 1]
    stats = analytics.get_stats()
    categories = ["New Visitors", "Returning Visits"]
    values = [stats["new_visitors"], stats["returning_visitors"]]
    colors = ["#22c55e", "#3b82f6"]
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("New vs Returning Visitors")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f"{val:,}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("analytics_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nDashboard saved to 'analytics_dashboard.png'")


# Create visualizations
visualize_analytics(analytics)
```

### Step 5: Accuracy Analysis

Let's verify our probabilistic structures are giving accurate results:

```python
from collections import Counter

def verify_accuracy(events: list[PageView], analytics: WebAnalytics):
    """Compare probabilistic estimates to exact counts."""

    # Calculate exact values
    exact_visitors = len(set(e.visitor_id for e in events))
    exact_page_counts = Counter(e.url for e in events)

    # Get estimates
    estimated_visitors = int(analytics.unique_visitors.cardinality())

    print("\n" + "=" * 50)
    print("ACCURACY VERIFICATION")
    print("=" * 50)

    # Visitor count accuracy
    visitor_error = abs(estimated_visitors - exact_visitors) / exact_visitors * 100
    print(f"\nUnique Visitors:")
    print(f"  Exact:     {exact_visitors:,}")
    print(f"  Estimated: {estimated_visitors:,}")
    print(f"  Error:     {visitor_error:.2f}%")

    # Page view accuracy (sample top pages)
    print(f"\nPage View Counts (top 5 pages):")
    print(f"  {'Page':<25} {'Exact':>10} {'Estimated':>10} {'Error':>8}")
    print(f"  {'-' * 55}")

    for url, exact in exact_page_counts.most_common(5):
        estimated = analytics.get_page_views(url)
        error = (estimated - exact) / exact * 100
        print(f"  {url:<25} {exact:>10,} {estimated:>10,} {error:>7.1f}%")

    # Memory comparison
    exact_memory = (
        exact_visitors * 50 +  # ~50 bytes per visitor ID in a set
        len(exact_page_counts) * 60  # ~60 bytes per URL in a dict
    )
    prob_memory = analytics.memory_usage()["total_bytes"]

    print(f"\nMemory Comparison:")
    print(f"  Exact data structures: ~{exact_memory / 1024 / 1024:.1f} MB")
    print(f"  Probabilistic:          {prob_memory / 1024 / 1024:.2f} MB")
    print(f"  Memory savings:         {(1 - prob_memory/exact_memory) * 100:.1f}%")


verify_accuracy(events, analytics)
```

## Complete Example

Here's the complete, runnable code:

```python
"""
Complete Web Analytics Example with Hazy

Run with: python web_analytics.py
"""

from hazy import BloomFilter, HyperLogLog, CountMinSketch, TopK
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from collections import Counter
import random


@dataclass
class PageView:
    visitor_id: str
    url: str
    timestamp: datetime
    referrer: Optional[str] = None


class WebAnalytics:
    def __init__(self, expected_visitors=1_000_000, top_k=100):
        self.unique_visitors = HyperLogLog(precision=14)
        self.page_views = CountMinSketch(width=10000, depth=5)
        self.trending = TopK(k=top_k)
        self.known_visitors = BloomFilter(
            expected_items=expected_visitors, false_positive_rate=0.01
        )
        self.total_pageviews = 0
        self.new_visitors = 0
        self.returning_visitors = 0

    def record(self, event: PageView):
        self.total_pageviews += 1
        self.unique_visitors.add(event.visitor_id)
        self.page_views.add(event.url)
        self.trending.add(event.url)

        if event.visitor_id in self.known_visitors:
            self.returning_visitors += 1
        else:
            self.new_visitors += 1
            self.known_visitors.add(event.visitor_id)

    def summary(self):
        print(f"\n{'='*50}")
        print("ANALYTICS SUMMARY")
        print(f"{'='*50}")
        print(f"Total Page Views:   {self.total_pageviews:,}")
        print(f"Unique Visitors:    {int(self.unique_visitors.cardinality()):,}")
        print(f"New Visitors:       {self.new_visitors:,}")
        print(f"Returning Visits:   {self.returning_visitors:,}")

        print(f"\nTop 10 Pages:")
        for url, count in self.trending.top(10):
            print(f"  {url:<30} {count:,}")

        total_mem = sum([
            self.unique_visitors.size_in_bytes,
            self.page_views.size_in_bytes,
            self.trending.size_in_bytes,
            self.known_visitors.size_in_bytes,
        ])
        print(f"\nTotal Memory: {total_mem / 1024:.1f} KB")


def main():
    # Setup
    analytics = WebAnalytics(expected_visitors=100_000)
    pages = ["/", "/about", "/products", "/blog", "/contact", "/pricing"]
    weights = [100, 20, 50, 40, 10, 30]

    # Simulate traffic
    print("Simulating 1 million page views...")
    visitors = [f"visitor_{i}" for i in range(100_000)]

    for _ in range(1_000_000):
        analytics.record(PageView(
            visitor_id=random.choice(visitors),
            url=random.choices(pages, weights=weights)[0],
            timestamp=datetime.now(),
        ))

    # Report
    analytics.summary()


if __name__ == "__main__":
    main()
```

## Key Takeaways

1. **HyperLogLog** is perfect for counting unique items — 16KB to count millions
2. **Count-Min Sketch** tracks frequencies without storing every URL
3. **TopK** finds heavy hitters with bounded memory
4. **BloomFilter** enables fast "have I seen this?" checks

## Exercises

1. **Add hourly breakdowns**: Create separate HLLs for each hour
2. **Track referrers**: Use another Count-Min Sketch for referrer frequencies
3. **Session tracking**: Use a Bloom filter to group page views into sessions
4. **A/B testing**: Track conversion rates for different page variants

## Next Tutorial

Continue to [Stream Deduplication](deduplication.md) to learn how to detect duplicates in real-time data streams.
