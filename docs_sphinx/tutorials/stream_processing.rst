Tutorial: Stream Processing
============================

In this tutorial, we'll build a stream processing pipeline that handles infinite event streams with bounded memory using multiple probabilistic data structures together.

The Problem
-----------

You're processing a continuous stream of events (logs, clicks, transactions, IoT data):

1. The stream is **infinite** — you can't store everything
2. You need **real-time answers** — can't wait for batch processing
3. Memory is **bounded** — can't grow indefinitely
4. You need multiple metrics: counts, uniques, frequencies, top items

Traditional approaches fail:

- Storing all events: runs out of memory
- Sampling: loses accuracy for rare events
- Batch processing: too slow for real-time

Solution Overview
-----------------

Combine multiple Hazy data structures for a complete stream processing solution:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Metric
     - Data Structure
     - What It Tells You
   * - Unique items
     - HyperLogLog
     - "How many distinct users/IPs/sessions?"
   * - Item frequency
     - CountMinSketch
     - "How many times did X occur?"
   * - Heavy hitters
     - TopK
     - "What are the most common items?"
   * - Set membership
     - BloomFilter
     - "Have I seen this item before?"
   * - Item similarity
     - MinHash
     - "Are these two sets similar?"

Implementation
--------------

Step 1: Build the Stream Processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hazy import (
       BloomFilter,
       HyperLogLog,
       CountMinSketch,
       TopK,
       MinHash,
   )
   from dataclasses import dataclass, field
   from datetime import datetime
   from typing import Any, Callable
   from collections import defaultdict


   @dataclass
   class StreamEvent:
       """A generic event in the stream."""
       event_type: str
       entity_id: str
       properties: dict = field(default_factory=dict)
       timestamp: datetime = field(default_factory=datetime.now)


   class StreamProcessor:
       """
       Real-time stream processor using probabilistic data structures.

       Processes infinite streams with bounded memory.
       """

       def __init__(
           self,
           expected_unique_entities: int = 1_000_000,
           top_k: int = 100,
       ):
           # Track unique entities (users, sessions, etc.)
           self.unique_entities = HyperLogLog(precision=14)

           # Track event type frequencies
           self.event_frequencies = CountMinSketch(width=10000, depth=5)

           # Track top entities by activity
           self.top_entities = TopK(k=top_k)

           # Track top event types
           self.top_event_types = TopK(k=50)

           # Seen entities (for first-time detection)
           self.seen_entities = BloomFilter(
               expected_items=expected_unique_entities,
               false_positive_rate=0.01
           )

           # Per-entity event signatures (for behavior analysis)
           self.entity_signatures: dict[str, MinHash] = {}

           # Metrics
           self.total_events = 0
           self.first_time_entities = 0
           self.start_time = datetime.now()

       def process(self, event: StreamEvent) -> dict:
           """
           Process a single event from the stream.

           Returns:
               Dict with event metadata and computed flags
           """
           self.total_events += 1
           result = {
               "event": event,
               "is_first_time": False,
               "entity_event_count": 0,
           }

           # Track unique entities
           self.unique_entities.add(event.entity_id)

           # Track event frequencies
           self.event_frequencies.add(event.event_type)
           self.top_event_types.add(event.event_type)

           # Track entity activity
           self.top_entities.add(event.entity_id)
           result["entity_event_count"] = self.event_frequencies[event.entity_id]

           # Check if first-time entity
           if event.entity_id not in self.seen_entities:
               self.first_time_entities += 1
               result["is_first_time"] = True
               self.seen_entities.add(event.entity_id)

           # Build entity behavior signature
           if event.entity_id not in self.entity_signatures:
               self.entity_signatures[event.entity_id] = MinHash(num_perm=128)
           self.entity_signatures[event.entity_id].add(event.event_type)

           return result

       def process_batch(self, events: list[StreamEvent]) -> list[dict]:
           """Process a batch of events."""
           return [self.process(event) for event in events]

       def get_stats(self) -> dict:
           """Get current stream statistics."""
           elapsed = (datetime.now() - self.start_time).total_seconds()
           return {
               "total_events": self.total_events,
               "unique_entities": int(self.unique_entities.cardinality()),
               "first_time_entities": self.first_time_entities,
               "events_per_second": self.total_events / max(elapsed, 1),
               "elapsed_seconds": elapsed,
           }

       def get_top_entities(self, n: int = 10) -> list:
           """Get most active entities."""
           return self.top_entities.top(n)

       def get_top_event_types(self, n: int = 10) -> list:
           """Get most common event types."""
           return self.top_event_types.top(n)

       def get_event_count(self, event_type: str) -> int:
           """Get count for a specific event type."""
           return self.event_frequencies[event_type]

       def get_similar_entities(
           self,
           entity_id: str,
           threshold: float = 0.5
       ) -> list[tuple[str, float]]:
           """Find entities with similar behavior patterns."""
           if entity_id not in self.entity_signatures:
               return []

           target_sig = self.entity_signatures[entity_id]
           similar = []

           for other_id, other_sig in self.entity_signatures.items():
               if other_id == entity_id:
                   continue
               similarity = target_sig.jaccard(other_sig)
               if similarity >= threshold:
                   similar.append((other_id, similarity))

           return sorted(similar, key=lambda x: -x[1])

       def memory_usage(self) -> dict:
           """Get memory usage breakdown."""
           sig_memory = sum(
               sig.size_in_bytes
               for sig in self.entity_signatures.values()
           )
           return {
               "unique_entities_hll": self.unique_entities.size_in_bytes,
               "event_frequencies_cms": self.event_frequencies.size_in_bytes,
               "top_entities_topk": self.top_entities.size_in_bytes,
               "top_events_topk": self.top_event_types.size_in_bytes,
               "seen_entities_bloom": self.seen_entities.size_in_bytes,
               "entity_signatures_minhash": sig_memory,
               "total_bytes": (
                   self.unique_entities.size_in_bytes +
                   self.event_frequencies.size_in_bytes +
                   self.top_entities.size_in_bytes +
                   self.top_event_types.size_in_bytes +
                   self.seen_entities.size_in_bytes +
                   sig_memory
               ),
           }

Step 2: Simulate an Event Stream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import random
   import time

   def generate_event_stream(
       n_events: int,
       n_entities: int = 10_000,
       event_types: list[str] = None
   ):
       """Generate a realistic event stream."""

       if event_types is None:
           event_types = [
               "page_view", "click", "scroll", "form_submit",
               "add_to_cart", "purchase", "search", "login",
               "logout", "share", "comment", "like"
           ]

       # Power law distribution for entities (some users much more active)
       entity_weights = [1.0 / (i + 1) ** 0.7 for i in range(n_entities)]

       # Event type weights (page_view most common)
       event_weights = [50, 30, 20, 10, 8, 3, 15, 5, 4, 6, 7, 12]

       entities = [f"entity_{i}" for i in range(n_entities)]

       for _ in range(n_events):
           entity = random.choices(entities, weights=entity_weights)[0]
           event_type = random.choices(event_types, weights=event_weights)[0]

           yield StreamEvent(
               event_type=event_type,
               entity_id=entity,
               properties={"value": random.random()},
           )


   # Process stream
   print("=" * 60)
   print("STREAM PROCESSING DEMO")
   print("=" * 60)

   processor = StreamProcessor(
       expected_unique_entities=10_000,
       top_k=50
   )

   # Process 100,000 events
   print("\nProcessing 100,000 events...")
   start = time.time()

   for event in generate_event_stream(100_000, n_entities=10_000):
       processor.process(event)

   elapsed = time.time() - start

   # Show results
   stats = processor.get_stats()
   print(f"\nProcessed in {elapsed:.2f}s ({stats['events_per_second']:,.0f} events/sec)")

Step 3: Analyze the Stream
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Stream statistics
   print("\n" + "=" * 60)
   print("STREAM STATISTICS")
   print("=" * 60)
   print(f"Total events:       {stats['total_events']:,}")
   print(f"Unique entities:    {stats['unique_entities']:,}")
   print(f"First-time events:  {stats['first_time_entities']:,}")

   # Top event types
   print("\n" + "-" * 40)
   print("TOP EVENT TYPES")
   print("-" * 40)
   for event_type, count in processor.get_top_event_types(10):
       pct = count / stats['total_events'] * 100
       bar = "#" * int(pct)
       print(f"  {event_type:<15} {count:>7,} ({pct:>5.1f}%) {bar}")

   # Top entities
   print("\n" + "-" * 40)
   print("MOST ACTIVE ENTITIES")
   print("-" * 40)
   for entity, count in processor.get_top_entities(10):
       print(f"  {entity:<15} {count:>6,} events")

   # Memory usage
   memory = processor.memory_usage()
   print("\n" + "-" * 40)
   print("MEMORY USAGE")
   print("-" * 40)
   print(f"  HyperLogLog:    {memory['unique_entities_hll']:>10,} bytes")
   print(f"  CountMinSketch: {memory['event_frequencies_cms']:>10,} bytes")
   print(f"  TopK (entities):{memory['top_entities_topk']:>10,} bytes")
   print(f"  TopK (events):  {memory['top_events_topk']:>10,} bytes")
   print(f"  BloomFilter:    {memory['seen_entities_bloom']:>10,} bytes")
   print(f"  MinHash sigs:   {memory['entity_signatures_minhash']:>10,} bytes")
   print(f"  TOTAL:          {memory['total_bytes']:>10,} bytes")
   print(f"                  ({memory['total_bytes'] / 1024 / 1024:.2f} MB)")

Step 4: Windowed Stream Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real systems often need time-windowed metrics:

.. code-block:: python

   from datetime import timedelta


   class WindowedStreamProcessor:
       """
       Stream processor with tumbling time windows.

       Maintains separate structures for each time window.
       """

       def __init__(
           self,
           window_duration: timedelta = timedelta(minutes=5),
           n_windows: int = 12  # Keep last hour of windows
       ):
           self.window_duration = window_duration
           self.n_windows = n_windows

           # Current window
           self.current_window_start = datetime.now()
           self.current_window = self._create_window()

           # Historical windows (circular buffer)
           self.windows: list[dict] = []

           # Global aggregates
           self.global_uniques = HyperLogLog(precision=14)
           self.total_events = 0

       def _create_window(self) -> dict:
           """Create a fresh window with all structures."""
           return {
               "start_time": datetime.now(),
               "events": 0,
               "uniques": HyperLogLog(precision=12),
               "top_items": TopK(k=20),
               "frequencies": CountMinSketch(width=5000, depth=4),
           }

       def _rotate_window_if_needed(self):
           """Check if current window has expired."""
           now = datetime.now()
           if now - self.current_window_start >= self.window_duration:
               # Save current window
               self.windows.append(self.current_window)
               if len(self.windows) > self.n_windows:
                   self.windows.pop(0)  # Remove oldest

               # Create new window
               self.current_window_start = now
               self.current_window = self._create_window()

       def process(self, event: StreamEvent):
           """Process an event."""
           self._rotate_window_if_needed()
           self.total_events += 1

           # Update current window
           self.current_window["events"] += 1
           self.current_window["uniques"].add(event.entity_id)
           self.current_window["top_items"].add(event.entity_id)
           self.current_window["frequencies"].add(event.event_type)

           # Update global
           self.global_uniques.add(event.entity_id)

       def get_current_window_stats(self) -> dict:
           """Get stats for current window."""
           return {
               "start_time": self.current_window["start_time"],
               "events": self.current_window["events"],
               "uniques": int(self.current_window["uniques"].cardinality()),
               "top_items": self.current_window["top_items"].top(5),
           }

       def get_historical_stats(self) -> list[dict]:
           """Get stats for all historical windows."""
           return [
               {
                   "start_time": w["start_time"],
                   "events": w["events"],
                   "uniques": int(w["uniques"].cardinality()),
               }
               for w in self.windows
           ]

       def get_hourly_uniques(self) -> int:
           """Estimate uniques over the last hour using HLL merge."""
           merged = HyperLogLog(precision=14)

           # Merge current window
           merged.merge(self.current_window["uniques"])

           # Merge historical windows
           for w in self.windows:
               merged.merge(w["uniques"])

           return int(merged.cardinality())


   # Demo windowed processing
   print("\n" + "=" * 60)
   print("WINDOWED STREAM PROCESSING")
   print("=" * 60)

   windowed_processor = WindowedStreamProcessor(
       window_duration=timedelta(seconds=2),  # Short for demo
       n_windows=5
   )

   # Process events across multiple windows
   for i, event in enumerate(generate_event_stream(10_000, n_entities=1000)):
       windowed_processor.process(event)

       # Force window rotation for demo
       if i % 2000 == 1999:
           windowed_processor.current_window_start -= timedelta(seconds=3)

   print(f"\nCurrent window: {windowed_processor.get_current_window_stats()}")
   print(f"Historical windows: {len(windowed_processor.windows)}")
   print(f"Hourly uniques (merged HLLs): {windowed_processor.get_hourly_uniques():,}")

Step 5: Anomaly Detection in Streams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use probabilistic structures to detect anomalies:

.. code-block:: python

   class StreamAnomalyDetector:
       """Detect anomalies in event streams."""

       def __init__(self):
           # Track "normal" event rates
           self.baseline_frequencies = CountMinSketch(width=10000, depth=5)
           self.baseline_events = 0

           # Track current period
           self.current_frequencies = CountMinSketch(width=10000, depth=5)
           self.current_events = 0

           # Track new entities (never seen before)
           self.known_entities = BloomFilter(
               expected_items=100_000,
               false_positive_rate=0.01
           )
           self.new_entity_count = 0

       def train(self, events: list[StreamEvent]):
           """Build baseline from historical events."""
           for event in events:
               self.baseline_frequencies.add(event.event_type)
               self.baseline_events += 1
               self.known_entities.add(event.entity_id)

       def check(self, event: StreamEvent) -> list[str]:
           """Check event for anomalies."""
           anomalies = []

           # Update current period
           self.current_frequencies.add(event.event_type)
           self.current_events += 1

           # Check 1: New entity
           if event.entity_id not in self.known_entities:
               self.new_entity_count += 1
               self.known_entities.add(event.entity_id)
               if self.new_entity_count > 100:  # Threshold
                   anomalies.append(f"High rate of new entities: {self.new_entity_count}")

           # Check 2: Event type frequency spike
           if self.current_events > 100:  # Need enough data
               baseline_rate = (
                   self.baseline_frequencies[event.event_type]
                   / max(self.baseline_events, 1)
               )
               current_rate = (
                   self.current_frequencies[event.event_type]
                   / self.current_events
               )

               if baseline_rate > 0 and current_rate > baseline_rate * 3:
                   anomalies.append(
                       f"Event '{event.event_type}' rate spike: "
                       f"{current_rate:.2%} vs baseline {baseline_rate:.2%}"
                   )

           return anomalies


   # Demo anomaly detection
   print("\n" + "=" * 60)
   print("ANOMALY DETECTION")
   print("=" * 60)

   detector = StreamAnomalyDetector()

   # Train on normal traffic
   print("\nTraining on normal traffic...")
   normal_events = list(generate_event_stream(10_000, n_entities=1000))
   detector.train(normal_events)

   # Process new traffic with anomalies injected
   print("Processing new traffic with anomalies...")
   anomaly_count = 0

   for i, event in enumerate(generate_event_stream(5_000, n_entities=2000)):
       # Inject anomaly: flood of 'purchase' events
       if 1000 <= i <= 1500:
           event.event_type = "purchase"

       anomalies = detector.check(event)
       if anomalies:
           anomaly_count += 1
           if anomaly_count <= 3:  # Show first few
               print(f"  Anomaly at event {i}: {anomalies}")

   print(f"\nTotal anomalies detected: {anomaly_count}")

Visualizing Stream Processing Results
-------------------------------------

Create a comprehensive dashboard to visualize the stream processing metrics:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Get data from the processor
   stats = processor.get_stats()
   memory = processor.memory_usage()
   top_events = processor.get_top_event_types(12)
   top_entities = processor.get_top_entities(10)

   fig, axes = plt.subplots(2, 3, figsize=(16, 10))
   fig.suptitle('Stream Processing Dashboard', fontsize=16, fontweight='bold')

   # 1. Event Type Distribution (pie chart)
   ax1 = axes[0, 0]
   event_labels = [e for e, _ in top_events[:6]]
   event_counts = [c for _, c in top_events[:6]]
   other_count = sum(c for _, c in top_events[6:])
   if other_count > 0:
       event_labels.append('Other')
       event_counts.append(other_count)
   colors = plt.cm.Set3(np.linspace(0, 1, len(event_labels)))
   ax1.pie(event_counts, labels=event_labels, colors=colors,
           autopct='%1.1f%%', startangle=90)
   ax1.set_title('Event Type Distribution')

   # 2. Top Entities Bar Chart
   ax2 = axes[0, 1]
   entities = [e for e, _ in top_entities]
   entity_counts = [c for _, c in top_entities]
   colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(entities)))
   bars = ax2.barh(entities, entity_counts, color=colors)
   ax2.set_xlabel('Event Count')
   ax2.set_title('Top 10 Most Active Entities')
   ax2.invert_yaxis()
   for bar, count in zip(bars, entity_counts):
       ax2.text(count + max(entity_counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=9)

   # 3. Memory Usage Breakdown
   ax3 = axes[0, 2]
   mem_components = ['HyperLogLog', 'CountMinSketch', 'TopK\n(entities)',
                     'TopK\n(events)', 'BloomFilter', 'MinHash\nSignatures']
   mem_sizes = [
       memory['unique_entities_hll'] / 1024,
       memory['event_frequencies_cms'] / 1024,
       memory['top_entities_topk'] / 1024,
       memory['top_events_topk'] / 1024,
       memory['seen_entities_bloom'] / 1024,
       memory['entity_signatures_minhash'] / 1024,
   ]
   colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
   bars = ax3.bar(mem_components, mem_sizes, color=colors)
   ax3.set_ylabel('Memory (KB)')
   ax3.set_title('Memory Usage by Component')
   ax3.tick_params(axis='x', rotation=45)
   for bar, size in zip(bars, mem_sizes):
       if size > 0:
           ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{size:.1f}', ha='center', fontsize=8)

   # 4. Throughput Over Time (simulated)
   ax4 = axes[1, 0]
   # Simulate throughput data
   time_points = np.arange(0, 10, 0.5)
   throughput = stats['events_per_second'] * (1 + 0.1 * np.sin(time_points))
   ax4.fill_between(time_points, throughput, alpha=0.3, color='#3498db')
   ax4.plot(time_points, throughput, color='#3498db', linewidth=2)
   ax4.axhline(y=stats['events_per_second'], color='red', linestyle='--',
               label=f"Avg: {stats['events_per_second']:,.0f}/s")
   ax4.set_xlabel('Time (seconds)')
   ax4.set_ylabel('Events/Second')
   ax4.set_title('Processing Throughput')
   ax4.legend()

   # 5. First-time vs Returning Events
   ax5 = axes[1, 1]
   first_time = stats['first_time_entities']
   returning = stats['total_events'] - first_time
   sizes = [first_time, returning]
   labels = [f'First-time\n({first_time:,})', f'Returning\n({returning:,})']
   colors = ['#2ecc71', '#3498db']
   explode = (0.05, 0)
   ax5.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
   ax5.set_title('First-time vs Returning Entities')

   # 6. Summary Statistics
   ax6 = axes[1, 2]
   ax6.axis('off')
   summary_text = f"""
   STREAM PROCESSING SUMMARY
   {'─' * 38}

   EVENTS PROCESSED:
   Total Events:          {stats['total_events']:>12,}
   Unique Entities:       {stats['unique_entities']:>12,}
   First-time Entities:   {stats['first_time_entities']:>12,}

   PERFORMANCE:
   Processing Time:       {stats['elapsed_seconds']:>11.2f}s
   Throughput:            {stats['events_per_second']:>10,.0f}/s

   {'─' * 38}
   MEMORY EFFICIENCY:
   Total Memory Used:     {memory['total_bytes']/1024:.1f} KB
                          ({memory['total_bytes']/1024/1024:.2f} MB)

   Events per KB:         {stats['total_events']/(memory['total_bytes']/1024):.0f}
   {'─' * 38}
   Bounded memory for infinite streams!
   """
   ax6.text(0.05, 0.5, summary_text, fontsize=10, fontfamily='monospace',
            verticalalignment='center', transform=ax6.transAxes,
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

   plt.tight_layout()
   plt.savefig('stream_processing_dashboard.png', dpi=150, bbox_inches='tight')
   plt.show()

This visualization provides a complete overview of the stream processing system: event distribution, top entities, memory usage, throughput, and key metrics.

.. image:: /_static/images/stream_processing_dashboard.png
   :alt: Stream Processing Dashboard
   :align: center
   :width: 100%

Key Takeaways
-------------

1. **Combine structures** for comprehensive stream analysis
2. **Bounded memory** — process infinite streams without growing memory
3. **HyperLogLog merge** enables aggregating across time windows
4. **Bloom filters** efficiently track "first time" events
5. **MinHash** finds similar behavior patterns across entities

Real-World Applications
-----------------------

- **Log analysis**: Process server logs in real-time
- **IoT data**: Handle sensor streams from millions of devices
- **Clickstream**: Analyze user behavior as it happens
- **Financial**: Detect fraud patterns in transaction streams
- **Network**: Monitor traffic patterns and detect anomalies

Exercises
---------

1. **Sliding windows**: Implement sliding (vs tumbling) windows
2. **Percentile estimation**: Add approximate percentile tracking
3. **Correlation detection**: Find event types that occur together
4. **Distributed processing**: Merge sketches from multiple processors

Conclusion
----------

You now have a complete toolkit for stream processing with bounded memory. These patterns are used by major systems like Apache Kafka Streams, Apache Flink, and Google Cloud Dataflow.

Check out the :doc:`../api/index` for the complete API reference.
