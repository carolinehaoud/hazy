Tutorial: Real-time Leaderboards
=================================

In this tutorial, we'll build a real-time trending/leaderboard system that tracks top items across massive event streams.

The Problem
-----------

You're building a system that needs to track "what's hot right now":

1. **Trending hashtags** on a social platform
2. **Top search queries** on a search engine
3. **Most played songs** on a music service
4. **Best-selling products** in an e-commerce store
5. **Most active users** in a gaming leaderboard

The naive approach — counting everything in a database — doesn't scale when you have millions of events per minute.

Solution Overview
-----------------

The **TopK** data structure is designed exactly for this problem:

- Tracks the K most frequent items in a stream
- Uses bounded memory regardless of stream size
- Provides approximate counts (slight overestimate possible)
- Handles high-throughput streams efficiently

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Use Case
     - K Value
     - Memory
     - Updates/sec
   * - Top 100 trending
     - 100
     - ~10 KB
     - Millions
   * - Top 1000 products
     - 1000
     - ~100 KB
     - Millions
   * - Top 10000 queries
     - 10000
     - ~1 MB
     - Millions

Implementation
--------------

Step 1: Basic Trending System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hazy import TopK, HyperLogLog, CountMinSketch
   from dataclasses import dataclass
   from datetime import datetime, timedelta
   from typing import Optional
   from collections import defaultdict


   @dataclass
   class Event:
       """A generic event in the stream."""
       item: str
       user_id: str
       timestamp: datetime
       metadata: Optional[dict] = None


   class TrendingTracker:
       """Track trending items in real-time."""

       def __init__(self, k: int = 100):
           # Track top K items globally
           self.global_trending = TopK(k=k)

           # Track unique users who engaged with each trending item
           # We'll use HyperLogLog for memory efficiency
           self.item_unique_users: dict[str, HyperLogLog] = {}

           # Total event count
           self.total_events = 0

       def record(self, event: Event):
           """Record an event."""
           self.total_events += 1

           # Update global trending
           self.global_trending.add(event.item)

           # Track unique users per item (for engagement metrics)
           if event.item not in self.item_unique_users:
               self.item_unique_users[event.item] = HyperLogLog(precision=10)
           self.item_unique_users[event.item].add(event.user_id)

       def get_trending(self, n: int = 10) -> list[tuple[str, int, int]]:
           """
           Get top N trending items.

           Returns:
               List of (item, count, unique_users)
           """
           top_items = self.global_trending.top(n)
           results = []

           for item, count in top_items:
               unique_users = 0
               if item in self.item_unique_users:
                   unique_users = int(self.item_unique_users[item].cardinality())
               results.append((item, count, unique_users))

           return results

       def get_stats(self) -> dict:
           """Get tracker statistics."""
           return {
               "total_events": self.total_events,
               "tracked_items": len(self.item_unique_users),
               "topk_memory": self.global_trending.size_in_bytes,
           }

Step 2: Multi-Window Trending
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real trending systems track items across multiple time windows:

.. code-block:: python

   class TimeWindowedTrending:
       """Track trending items across multiple time windows."""

       def __init__(self, k: int = 100):
           self.k = k

           # Different time windows
           self.windows = {
               "1min": TopK(k=k),
               "5min": TopK(k=k),
               "1hour": TopK(k=k),
               "24hour": TopK(k=k),
           }

           # Timestamps for window rotation
           self.window_starts = {
               "1min": datetime.now(),
               "5min": datetime.now(),
               "1hour": datetime.now(),
               "24hour": datetime.now(),
           }

           # Previous windows (for smooth transitions)
           self.previous_windows = {
               "1min": TopK(k=k),
               "5min": TopK(k=k),
               "1hour": TopK(k=k),
               "24hour": TopK(k=k),
           }

           # Window durations
           self.durations = {
               "1min": timedelta(minutes=1),
               "5min": timedelta(minutes=5),
               "1hour": timedelta(hours=1),
               "24hour": timedelta(hours=24),
           }

       def _rotate_if_needed(self, window_name: str, now: datetime):
           """Rotate a window if it's expired."""
           if now - self.window_starts[window_name] >= self.durations[window_name]:
               # Move current to previous
               self.previous_windows[window_name] = self.windows[window_name]
               # Create fresh window
               self.windows[window_name] = TopK(k=self.k)
               self.window_starts[window_name] = now

       def record(self, item: str, timestamp: Optional[datetime] = None):
           """Record an item occurrence."""
           now = timestamp or datetime.now()

           # Check for window rotations
           for window_name in self.windows:
               self._rotate_if_needed(window_name, now)

           # Add to all windows
           for window in self.windows.values():
               window.add(item)

       def get_trending(self, window: str = "1hour", n: int = 10) -> list:
           """Get trending items for a specific time window."""
           if window not in self.windows:
               raise ValueError(f"Unknown window: {window}")

           return self.windows[window].top(n)

       def get_rising(self, n: int = 10) -> list:
           """
           Get items that are rising fast.

           Compares 1-minute window to 1-hour window to find
           items with sudden popularity spikes.
           """
           recent = dict(self.windows["1min"].top(n * 3))
           hourly = dict(self.windows["1hour"].top(n * 10))

           rising = []
           for item, recent_count in recent.items():
               hourly_count = hourly.get(item, 0)
               if hourly_count > 0:
                   # Calculate velocity (recent activity vs historical)
                   velocity = (recent_count * 60) / hourly_count
                   rising.append((item, recent_count, velocity))

           # Sort by velocity
           rising.sort(key=lambda x: x[2], reverse=True)
           return rising[:n]

Step 3: Simulate a Social Media Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import random

   def simulate_social_media(n_events: int = 100_000):
       """Simulate hashtag usage on a social platform."""

       # Some hashtags are consistently popular
       evergreen_tags = ["#python", "#coding", "#tech", "#ai", "#ml"]

       # Some hashtags are trending right now
       trending_tags = ["#breaking", "#viral", "#trending"]

       # Most hashtags have low usage
       random_tags = [f"#topic{i}" for i in range(1000)]

       # Create weighted distribution (Zipf-like)
       all_tags = (
           evergreen_tags * 100 +    # Evergreen: high weight
           trending_tags * 500 +     # Trending: very high weight (current spike)
           random_tags               # Random: low weight
       )

       # Create tracker
       tracker = TimeWindowedTrending(k=50)

       # Generate events
       print(f"Simulating {n_events:,} social media events...")
       users = [f"user_{i}" for i in range(10_000)]

       for i in range(n_events):
           hashtag = random.choice(all_tags)
           tracker.record(hashtag)

           # Progress update
           if (i + 1) % 25_000 == 0:
               print(f"  Processed {i + 1:,} events...")

       return tracker


   # Run simulation
   tracker = simulate_social_media(100_000)

   # Show results
   print("\n" + "=" * 60)
   print("TRENDING HASHTAGS (1 Hour Window)")
   print("=" * 60)
   for i, (tag, count) in enumerate(tracker.get_trending("1hour", 15), 1):
       bar = "#" * min(count // 100, 40)
       print(f"{i:>2}. {tag:<20} {count:>6,} {bar}")

   print("\n" + "=" * 60)
   print("RISING FAST (Velocity Analysis)")
   print("=" * 60)
   for i, (tag, count, velocity) in enumerate(tracker.get_rising(10), 1):
       print(f"{i:>2}. {tag:<20} {count:>5,} recent | {velocity:.1f}x velocity")

Step 4: Gaming Leaderboard
~~~~~~~~~~~~~~~~~~~~~~~~~~

TopK also works great for gaming leaderboards:

.. code-block:: python

   class GamingLeaderboard:
       """Real-time gaming leaderboard."""

       def __init__(self, top_n: int = 100):
           # Track top players by score
           self.leaderboard = TopK(k=top_n)

           # Track total games per player
           self.games_played = CountMinSketch(width=10000, depth=5)

           # Track unique active players
           self.active_players = HyperLogLog(precision=14)

       def record_game(self, player_id: str, score: int):
           """Record a game result."""
           # Add score to leaderboard
           # Note: TopK tracks frequency, so we add multiple times for score
           for _ in range(score):
               self.leaderboard.add(player_id)

           # Track games played
           self.games_played.add(player_id)

           # Track active players
           self.active_players.add(player_id)

       def record_score(self, player_id: str, score: int):
           """Alternative: directly add with count."""
           self.leaderboard.add(player_id, count=score)
           self.games_played.add(player_id)
           self.active_players.add(player_id)

       def get_leaderboard(self, n: int = 10) -> list:
           """Get top N players."""
           return self.leaderboard.top(n)

       def get_player_stats(self, player_id: str) -> dict:
           """Get stats for a specific player."""
           # Check if player is in top K
           top = dict(self.leaderboard.top(self.leaderboard.k))
           rank = None
           if player_id in top:
               sorted_players = sorted(top.items(), key=lambda x: -x[1])
               for i, (pid, _) in enumerate(sorted_players, 1):
                   if pid == player_id:
                       rank = i
                       break

           return {
               "player_id": player_id,
               "score": top.get(player_id, 0),
               "rank": rank,
               "games_played": self.games_played[player_id],
           }

       def get_stats(self) -> dict:
           """Get overall leaderboard stats."""
           return {
               "active_players": int(self.active_players.cardinality()),
               "memory_bytes": (
                   self.leaderboard.size_in_bytes +
                   self.games_played.size_in_bytes +
                   self.active_players.size_in_bytes
               ),
           }


   # Simulate gaming sessions
   print("\n" + "=" * 60)
   print("GAMING LEADERBOARD SIMULATION")
   print("=" * 60)

   leaderboard = GamingLeaderboard(top_n=50)

   # Simulate 10,000 game sessions
   players = [f"player_{i}" for i in range(1000)]

   # Some players are much better than others
   skill_levels = {p: random.gauss(100, 30) for p in players}

   for _ in range(10_000):
       player = random.choice(players)
       base_score = max(0, int(skill_levels[player] + random.gauss(0, 20)))
       leaderboard.record_score(player, base_score)

   # Show leaderboard
   print("\nTOP 10 PLAYERS:")
   for i, (player, score) in enumerate(leaderboard.get_leaderboard(10), 1):
       print(f"{i:>2}. {player:<15} Score: {score:>8,}")

   stats = leaderboard.get_stats()
   print(f"\nActive players: {stats['active_players']:,}")
   print(f"Memory used: {stats['memory_bytes']:,} bytes")

Visualizing the Leaderboard
---------------------------

Create a visualization of the trending and leaderboard data:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Create visualization
   fig, axes = plt.subplots(2, 2, figsize=(14, 10))
   fig.suptitle('Real-time Leaderboard Dashboard', fontsize=16, fontweight='bold')

   # 1. Trending Hashtags Bar Chart
   ax1 = axes[0, 0]
   trending_data = tracker.get_trending("1hour", 10)
   tags = [tag for tag, _ in trending_data]
   counts = [count for _, count in trending_data]
   colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(tags)))
   bars = ax1.barh(tags, counts, color=colors)
   ax1.set_xlabel('Event Count')
   ax1.set_title('Top 10 Trending Hashtags')
   ax1.invert_yaxis()
   for bar, count in zip(bars, counts):
       ax1.text(count + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=9)

   # 2. Gaming Leaderboard
   ax2 = axes[0, 1]
   top_players = leaderboard.get_leaderboard(10)
   players = [p for p, _ in top_players]
   scores = [s for _, s in top_players]
   colors = ['#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32' if i == 2
             else '#3498db' for i in range(len(players))]
   bars = ax2.barh(players, scores, color=colors)
   ax2.set_xlabel('Total Score')
   ax2.set_title('Gaming Leaderboard - Top 10')
   ax2.invert_yaxis()
   for i, (bar, score) in enumerate(zip(bars, scores)):
       medal = ['🥇', '🥈', '🥉'][i] if i < 3 else ''
       ax2.text(score + max(scores)*0.01, bar.get_y() + bar.get_height()/2,
                f'{medal} {score:,}', va='center', fontsize=9)

   # 3. Hashtag Popularity Distribution
   ax3 = axes[1, 0]
   all_trending = tracker.get_trending("1hour", 50)
   positions = range(1, len(all_trending) + 1)
   all_counts = [count for _, count in all_trending]
   ax3.scatter(positions, all_counts, c=all_counts, cmap='YlOrRd', s=100, alpha=0.7)
   ax3.plot(positions, all_counts, 'b-', alpha=0.3)
   ax3.set_xlabel('Rank')
   ax3.set_ylabel('Event Count')
   ax3.set_title('Popularity Distribution (Zipf-like pattern)')
   ax3.set_yscale('log')

   # 4. Summary Statistics
   ax4 = axes[1, 1]
   ax4.axis('off')

   game_stats = leaderboard.get_stats()
   summary_text = f"""
   LEADERBOARD STATISTICS
   {'─' * 40}

   TRENDING HASHTAGS:
   Total Events:          {100_000:>12,}
   Tracked Tags:                    50
   Top Tag Events:        {counts[0]:>12,}

   GAMING LEADERBOARD:
   Active Players:        {game_stats['active_players']:>12,}
   Top Score:             {scores[0]:>12,}
   Score Gap (1st-2nd):   {scores[0]-scores[1]:>12,}

   {'─' * 40}
   MEMORY EFFICIENCY:
   TopK Memory:           {game_stats['memory_bytes']:>10,} bytes
   Players Tracked:                 50
   Bytes per Player:      {game_stats['memory_bytes']//50:>10,}

   Time Complexity: O(log k) per update
   """
   ax4.text(0.05, 0.5, summary_text, fontsize=11, fontfamily='monospace',
            verticalalignment='center', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

   plt.tight_layout()
   plt.savefig('leaderboard_dashboard.png', dpi=150, bbox_inches='tight')
   plt.show()

This visualization displays trending hashtags, gaming leaderboard rankings, the Zipf-like popularity distribution, and system statistics.

Key Takeaways
-------------

1. **TopK** is perfect for finding heavy hitters in streams
2. **Bounded memory** — tracks top K regardless of total items
3. **Combine with HyperLogLog** for unique engagement metrics
4. **Time windows** enable "rising" and velocity detection
5. **Works for any ranking** — hashtags, products, players, queries

Exercises
---------

1. **Decay over time**: Implement exponential decay for older events
2. **Category trending**: Track top items per category using multiple TopK instances
3. **Personalized trending**: Combine global trending with user preferences
4. **Anomaly detection**: Alert when an item's velocity exceeds a threshold

Next Tutorial
-------------

Continue to :doc:`database_optimization` to learn how databases use Bloom filters for query optimization.
