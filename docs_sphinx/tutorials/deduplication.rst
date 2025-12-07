Tutorial: Stream Deduplication
==============================

Learn how to detect and filter duplicate items in real-time data streams using Bloom filters and related structures.

The Problem
-----------

You're processing a stream of events (URLs, transaction IDs, log entries) and need to:

1. **Detect duplicates** in real-time
2. **Use bounded memory** regardless of stream size
3. **Handle high throughput** with low latency
4. **Accept occasional false positives** (marking new items as duplicates)

Use Cases
---------

- **Web crawler**: Don't re-crawl URLs you've already visited
- **Event processing**: Ensure exactly-once semantics
- **Log deduplication**: Filter repeated log entries
- **Ad deduplication**: Don't show the same ad twice

Solution: Bloom Filter Deduplication
------------------------------------

.. code-block:: python

   from hazy import BloomFilter, ScalableBloomFilter
   from dataclasses import dataclass
   from datetime import datetime
   from typing import Iterator, Callable


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

Example: URL Deduplication for Web Crawler
------------------------------------------

.. code-block:: python

   import random
   import time

   def generate_urls(n_urls: int, duplicate_rate: float = 0.3) -> Iterator[Event]:
       """Generate a stream of URLs with some duplicates."""
       domains = ["example.com", "test.org", "sample.net", "demo.io"]
       paths = ["/", "/about", "/products", "/blog", "/contact"]

       seen_urls = []

       for i in range(n_urls):
           if seen_urls and random.random() < duplicate_rate:
               url = random.choice(seen_urls)
           else:
               domain = random.choice(domains)
               path = random.choice(paths)
               url = f"https://{domain}{path}"
               seen_urls.append(url)
               if len(seen_urls) > 10000:
                   seen_urls = seen_urls[-5000:]

           yield Event(id=url, payload={"url": url})


   # Create deduplicator
   dedup = StreamDeduplicator(
       expected_items=100_000,
       false_positive_rate=0.001
   )

   # Track unique URLs
   unique_urls = []

   def mock_crawl(event: Event):
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
   print(f"Total URLs processed:  {stats['total_processed']:,}")
   print(f"Unique URLs (crawled): {stats['unique']:,}")
   print(f"Duplicates filtered:   {stats['duplicates']:,}")
   print(f"Throughput:            {stats['total_processed']/elapsed:,.0f} URLs/sec")

Advanced: Time-Windowed Deduplication
-------------------------------------

For streams where you only care about recent duplicates:

.. code-block:: python

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
           self.window_minutes = window_minutes
           self.bucket_minutes = window_minutes // buckets
           self.expected_items = expected_items_per_bucket
           self.fpr = false_positive_rate

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
           """Process an event, returning True if new within the time window."""
           self._maybe_rotate()
           self.stats["processed"] += 1

           if self.is_duplicate(event_id):
               self.stats["duplicates"] += 1
               return False

           self.buckets[-1].add(event_id)
           self.stats["unique"] += 1
           return True

Deduplication with Deletion: Cuckoo Filter
-------------------------------------------

When you need to **remove** items (e.g., allow re-processing after some time):

.. code-block:: python

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

           self.seen.add(event_id)
           self.stats["unique"] += 1
           return True

       def remove(self, event_id: str) -> bool:
           """Remove an event, allowing it to be processed again."""
           if event_id in self.seen:
               self.seen.remove(event_id)
               self.stats["removed"] += 1
               return True
           return False


   # Example usage
   dedup = DeletableDeduplicator(capacity=10_000)

   # Process events
   for i in range(1000):
       dedup.process(f"event_{i}")

   print(f"After processing: {dedup.stats['unique']} unique events")

   # Remove some to allow re-processing
   for i in range(100):
       dedup.remove(f"event_{i}")

   print(f"After removal: {dedup.stats['removed']} can be re-processed")

Best Practices
--------------

1. **Choose the Right Structure**

   - Simple deduplication: ``BloomFilter``
   - Unknown stream size: ``ScalableBloomFilter``
   - Need to remove items: ``CuckooFilter``
   - Time-windowed: Rotating ``BloomFilter``\s

2. **Size Appropriately**

   .. code-block:: python

      # Rule of thumb: size for 2x expected items
      bf = BloomFilter(
          expected_items=expected_unique * 2,
          false_positive_rate=0.01
      )

3. **Monitor Fill Ratio**

   .. code-block:: python

      if bf.fill_ratio > 0.5:
          print("Warning: Filter getting full, FPR increasing")

4. **Consider Persistence**

   .. code-block:: python

      # Save filter state for recovery
      dedup.seen.save("dedup_state.hazy")

      # Restore on restart
      dedup.seen = BloomFilter.load("dedup_state.hazy")

Next Tutorial
-------------

Continue to :doc:`similarity_search` to learn how to find similar documents using MinHash.
