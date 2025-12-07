Tutorial: Web Analytics Dashboard
=================================

In this tutorial, we'll build a memory-efficient web analytics system that tracks unique visitors, page views, and trending content using probabilistic data structures.

The Problem
-----------

You're building analytics for a website with millions of page views per day. You need to track:

1. **Unique visitors** per day (cardinality)
2. **Page view counts** for each URL (frequency)
3. **Trending pages** (top-k)
4. **Returning vs new visitors** (set membership)

Storing every event exactly would require gigabytes of memory. Instead, we'll use probabilistic structures that give us accurate-enough answers with megabytes.

Solution Overview
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 15 35

   * - Metric
     - Data Structure
     - Memory
     - Accuracy
   * - Unique visitors
     - HyperLogLog
     - 16 KB
     - ~1% error
   * - Page view counts
     - Count-Min Sketch
     - 400 KB
     - Slight overcount
   * - Trending pages
     - TopK
     - ~10 KB
     - Exact for heavy hitters
   * - Returning visitors
     - BloomFilter
     - ~1.2 MB
     - 1% false positive

**Total: ~1.6 MB** to track millions of events!

Implementation
--------------

Step 1: Set Up the Analytics Tracker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hazy import BloomFilter, HyperLogLog, CountMinSketch, TopK
   from datetime import datetime
   from dataclasses import dataclass
   from typing import Optional


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

Step 2: Simulate Traffic
~~~~~~~~~~~~~~~~~~~~~~~~

Let's generate realistic web traffic to test our analytics:

.. code-block:: python

   import random

   def generate_traffic(n_events: int, n_unique_visitors: int) -> list[PageView]:
       """Generate simulated web traffic."""

       # Popular pages (Zipf distribution - few pages get most traffic)
       pages = [
           "/",
           "/about",
           "/products",
           "/products/widget-a",
           "/products/widget-b",
           "/blog",
           "/blog/post-1",
           "/blog/post-2",
           "/contact",
           "/pricing",
       ]

       # Page popularity weights (home page most popular)
       weights = [100, 20, 50, 30, 25, 40, 15, 12, 8, 25]

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

Step 3: Process Events and Analyze
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create analytics tracker
   analytics = WebAnalytics(
       expected_daily_visitors=100_000,
       expected_daily_pageviews=1_000_000
   )

   # Process all events
   print("Processing events...")
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
   print(f"  TOTAL:                   {memory['total_bytes']:>10,} bytes")
   print(f"                           ({memory['total_bytes'] / 1024 / 1024:.2f} MB)")

Step 4: Visualize the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a dashboard visualization of your analytics:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Create a 2x2 dashboard
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   fig.suptitle('Web Analytics Dashboard', fontsize=16, fontweight='bold')

   # 1. Trending Pages (horizontal bar chart)
   ax1 = axes[0, 0]
   trending = analytics.get_trending(10)
   urls = [url for url, _ in trending]
   counts = [count for _, count in trending]
   colors = plt.cm.Blues([(c / max(counts)) * 0.7 + 0.3 for c in counts])
   ax1.barh(urls, counts, color=colors)
   ax1.set_xlabel('Page Views')
   ax1.set_title('Top 10 Trending Pages')
   ax1.invert_yaxis()
   for i, (url, count) in enumerate(trending):
       ax1.text(count + max(counts) * 0.01, i, f'{count:,}', va='center', fontsize=9)

   # 2. New vs Returning Visitors (pie chart)
   ax2 = axes[0, 1]
   stats = analytics.get_stats()
   visitor_data = [stats['new_visitors'], stats['returning_visitors']]
   labels = ['New Visitors', 'Returning Visitors']
   colors = ['#66b3ff', '#99ff99']
   explode = (0.05, 0)
   ax2.pie(visitor_data, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
   ax2.set_title('New vs Returning Visitors')

   # 3. Memory Usage by Component (bar chart)
   ax3 = axes[1, 0]
   memory = analytics.memory_usage()
   components = ['HyperLogLog', 'Count-Min\nSketch', 'TopK', 'BloomFilter']
   sizes_kb = [
       memory['unique_visitors_hll'] / 1024,
       memory['page_views_cms'] / 1024,
       memory['trending_topk'] / 1024,
       memory['known_visitors_bloom'] / 1024
   ]
   colors = ['#ff9999', '#ffcc99', '#99ccff', '#cc99ff']
   bars = ax3.bar(components, sizes_kb, color=colors)
   ax3.set_ylabel('Memory (KB)')
   ax3.set_title('Memory Usage by Component')
   for bar, size in zip(bars, sizes_kb):
       ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{size:.1f} KB', ha='center', fontsize=9)

   # 4. Key Metrics Summary (text display)
   ax4 = axes[1, 1]
   ax4.axis('off')
   metrics_text = f"""
   KEY METRICS SUMMARY
   {'─' * 35}

   Total Page Views:     {stats['total_pageviews']:>12,}
   Unique Visitors:      {stats['unique_visitors']:>12,}
   Pages per Visitor:    {stats['pages_per_visitor']:>12.2f}

   Total Memory Used:    {memory['total_bytes'] / 1024 / 1024:>10.2f} MB

   {'─' * 35}
   Memory Savings vs Exact Storage:
   ~{(stats['total_pageviews'] * 100) / memory['total_bytes']:.0f}x compression!
   """
   ax4.text(0.1, 0.5, metrics_text, fontsize=12, fontfamily='monospace',
            verticalalignment='center', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

   plt.tight_layout()
   plt.savefig('web_analytics_dashboard.png', dpi=150, bbox_inches='tight')
   plt.show()

This produces a comprehensive dashboard showing trending pages, visitor breakdown, memory efficiency, and key metrics at a glance.

Key Takeaways
-------------

1. **HyperLogLog** is perfect for counting unique items — 16KB to count millions
2. **Count-Min Sketch** tracks frequencies without storing every URL
3. **TopK** finds heavy hitters with bounded memory
4. **BloomFilter** enables fast "have I seen this?" checks

Exercises
---------

1. **Add hourly breakdowns**: Create separate HLLs for each hour
2. **Track referrers**: Use another Count-Min Sketch for referrer frequencies
3. **Session tracking**: Use a Bloom filter to group page views into sessions
4. **A/B testing**: Track conversion rates for different page variants

Next Tutorial
-------------

Continue to :doc:`deduplication` to learn how to detect duplicates in real-time data streams.
