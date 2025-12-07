Tutorial: Spam & Fraud Detection
=================================

In this tutorial, we'll build a real-time fraud detection system that identifies suspicious actors using probabilistic data structures.

The Problem
-----------

You're building a payment or messaging system that needs to block known bad actors:

1. **Known bad emails** - block spam or fraudulent accounts
2. **Suspicious IP addresses** - rate limit or block malicious IPs
3. **Stolen credit cards** - check against known compromised cards
4. **Device fingerprints** - track repeat offenders across accounts

Storing millions of blocklist entries in a database and querying it for every request is slow. We need sub-millisecond lookups that scale.

Solution Overview
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 20 30

   * - Check
     - Data Structure
     - Memory
     - Tradeoff
   * - Email blocklist
     - BloomFilter
     - ~1.2 MB
     - 0.1% false positives OK (extra review)
   * - IP reputation
     - CuckooFilter
     - ~2 MB
     - Need deletion support
   * - Card blocklist
     - BloomFilter
     - ~600 KB
     - 0.01% false positives
   * - Device tracking
     - CountMinSketch
     - ~400 KB
     - Count fraud attempts

**Key insight**: False positives are acceptable — they just trigger additional verification. False negatives (missing actual fraud) are what we want to minimize.

Implementation
--------------

Step 1: Set Up the Fraud Detection System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hazy import BloomFilter, CuckooFilter, CountMinSketch
   from dataclasses import dataclass
   from typing import Optional
   from enum import Enum


   class RiskLevel(Enum):
       LOW = "low"
       MEDIUM = "medium"
       HIGH = "high"
       BLOCKED = "blocked"


   @dataclass
   class Transaction:
       """Represents a transaction to check."""
       user_id: str
       email: str
       ip_address: str
       card_hash: str  # Never store actual card numbers!
       device_fingerprint: str
       amount: float


   class FraudDetector:
       """Real-time fraud detection using probabilistic data structures."""

       def __init__(
           self,
           expected_bad_emails: int = 1_000_000,
           expected_bad_ips: int = 500_000,
           expected_bad_cards: int = 500_000,
       ):
           # Email blocklist - very low false positive rate
           # BloomFilter: fast, memory-efficient, no deletion
           self.bad_emails = BloomFilter(
               expected_items=expected_bad_emails,
               false_positive_rate=0.001  # 0.1% - extra review is cheap
           )

           # IP reputation - using CuckooFilter for deletion support
           # IPs get rehabilitated over time, so we need to remove them
           self.suspicious_ips = CuckooFilter(
               capacity=expected_bad_ips,
               fingerprint_size=16  # 16-bit fingerprints
           )

           # Stolen/compromised card blocklist
           # Very low false positive rate - blocking a good card is costly
           self.blocked_cards = BloomFilter(
               expected_items=expected_bad_cards,
               false_positive_rate=0.0001  # 0.01%
           )

           # Track fraud attempts per device fingerprint
           # Helps identify repeat offenders across accounts
           self.device_attempts = CountMinSketch(width=10000, depth=5)

           # Thresholds
           self.device_attempt_threshold = 5

       def add_bad_email(self, email: str):
           """Add email to blocklist."""
           self.bad_emails.add(email.lower().strip())

       def add_suspicious_ip(self, ip: str):
           """Add IP to suspicious list."""
           self.suspicious_ips.add(ip)

       def remove_suspicious_ip(self, ip: str):
           """Remove IP from suspicious list (rehabilitated)."""
           self.suspicious_ips.remove(ip)

       def add_blocked_card(self, card_hash: str):
           """Add card hash to blocklist."""
           self.blocked_cards.add(card_hash)

       def record_fraud_attempt(self, device_fingerprint: str):
           """Record a fraud attempt from a device."""
           self.device_attempts.add(device_fingerprint)

       def check_transaction(self, txn: Transaction) -> tuple[RiskLevel, list[str]]:
           """
           Check a transaction for fraud indicators.

           Returns:
               Tuple of (risk_level, list of reasons)
           """
           reasons = []
           risk_score = 0

           # Check email blocklist
           if txn.email.lower().strip() in self.bad_emails:
               reasons.append("Email on blocklist")
               risk_score += 100  # Immediate block

           # Check IP reputation
           if txn.ip_address in self.suspicious_ips:
               reasons.append("Suspicious IP address")
               risk_score += 50

           # Check card blocklist
           if txn.card_hash in self.blocked_cards:
               reasons.append("Card reported compromised")
               risk_score += 100  # Immediate block

           # Check device reputation
           device_attempts = self.device_attempts[txn.device_fingerprint]
           if device_attempts >= self.device_attempt_threshold:
               reasons.append(f"Device has {device_attempts} prior fraud attempts")
               risk_score += 30

           # Determine risk level
           if risk_score >= 100:
               return RiskLevel.BLOCKED, reasons
           elif risk_score >= 50:
               return RiskLevel.HIGH, reasons
           elif risk_score >= 20:
               return RiskLevel.MEDIUM, reasons
           else:
               return RiskLevel.LOW, reasons

       def memory_usage(self) -> dict:
           """Get memory usage breakdown."""
           return {
               "bad_emails": self.bad_emails.size_in_bytes,
               "suspicious_ips": self.suspicious_ips.size_in_bytes,
               "blocked_cards": self.blocked_cards.size_in_bytes,
               "device_attempts": self.device_attempts.size_in_bytes,
               "total_bytes": (
                   self.bad_emails.size_in_bytes +
                   self.suspicious_ips.size_in_bytes +
                   self.blocked_cards.size_in_bytes +
                   self.device_attempts.size_in_bytes
               ),
           }

Step 2: Load Blocklists
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import random
   import hashlib

   # Create detector
   detector = FraudDetector()

   # Simulate loading blocklists
   print("Loading blocklists...")

   # Add known bad emails
   bad_email_domains = ["spam.com", "fraud.net", "scam.org", "fake.io"]
   for i in range(100_000):
       domain = random.choice(bad_email_domains)
       email = f"user{i}@{domain}"
       detector.add_bad_email(email)

   # Add suspicious IPs (some IP ranges)
   for i in range(50_000):
       ip = f"10.0.{random.randint(0, 255)}.{random.randint(0, 255)}"
       detector.add_suspicious_ip(ip)

   # Add blocked card hashes
   for i in range(50_000):
       # In reality, this would be hashes of actual compromised cards
       card_hash = hashlib.sha256(f"stolen_card_{i}".encode()).hexdigest()
       detector.add_blocked_card(card_hash)

   print(f"Loaded blocklists")
   memory = detector.memory_usage()
   print(f"Total memory: {memory['total_bytes'] / 1024 / 1024:.2f} MB")

Step 3: Check Transactions in Real-Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_test_transactions():
       """Create test transactions - mix of good and bad."""
       transactions = []

       # Good transactions
       for i in range(1000):
           transactions.append(Transaction(
               user_id=f"good_user_{i}",
               email=f"legitimate{i}@gmail.com",
               ip_address=f"192.168.1.{i % 256}",
               card_hash=hashlib.sha256(f"good_card_{i}".encode()).hexdigest(),
               device_fingerprint=f"device_{i}",
               amount=random.uniform(10, 500),
           ))

       # Transactions from bad emails
       for i in range(100):
           transactions.append(Transaction(
               user_id=f"bad_email_user_{i}",
               email=f"user{i}@spam.com",
               ip_address=f"192.168.2.{i}",
               card_hash=hashlib.sha256(f"some_card_{i}".encode()).hexdigest(),
               device_fingerprint=f"bad_email_device_{i}",
               amount=random.uniform(100, 1000),
           ))

       # Transactions from suspicious IPs
       for i in range(100):
           transactions.append(Transaction(
               user_id=f"sus_ip_user_{i}",
               email=f"user{i}@yahoo.com",
               ip_address=f"10.0.{random.randint(0, 255)}.{random.randint(0, 255)}",
               card_hash=hashlib.sha256(f"another_card_{i}".encode()).hexdigest(),
               device_fingerprint=f"sus_ip_device_{i}",
               amount=random.uniform(50, 300),
           ))

       # Transactions with stolen cards
       for i in range(50):
           transactions.append(Transaction(
               user_id=f"stolen_card_user_{i}",
               email=f"innocent{i}@gmail.com",
               ip_address=f"192.168.3.{i}",
               card_hash=hashlib.sha256(f"stolen_card_{i}".encode()).hexdigest(),
               device_fingerprint=f"stolen_card_device_{i}",
               amount=random.uniform(500, 2000),
           ))

       random.shuffle(transactions)
       return transactions


   # Process transactions
   transactions = create_test_transactions()

   results = {
       RiskLevel.LOW: 0,
       RiskLevel.MEDIUM: 0,
       RiskLevel.HIGH: 0,
       RiskLevel.BLOCKED: 0,
   }

   print("\nProcessing transactions...")
   for txn in transactions:
       risk, reasons = detector.check_transaction(txn)
       results[risk] += 1

       # Log high-risk transactions
       if risk in [RiskLevel.HIGH, RiskLevel.BLOCKED]:
           # In production, you'd log this for review
           pass

   # Summary
   print("\n" + "=" * 50)
   print("TRANSACTION RESULTS")
   print("=" * 50)
   print(f"  Low Risk:     {results[RiskLevel.LOW]:>5} transactions")
   print(f"  Medium Risk:  {results[RiskLevel.MEDIUM]:>5} transactions")
   print(f"  High Risk:    {results[RiskLevel.HIGH]:>5} transactions")
   print(f"  Blocked:      {results[RiskLevel.BLOCKED]:>5} transactions")

Step 4: Handle IP Rehabilitation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One advantage of CuckooFilter over BloomFilter is deletion support:

.. code-block:: python

   # IP was flagged but investigation showed it's legitimate
   rehabilitated_ip = "10.0.50.100"

   # Add it first (simulate it was suspicious)
   detector.add_suspicious_ip(rehabilitated_ip)
   print(f"Is {rehabilitated_ip} suspicious? {rehabilitated_ip in detector.suspicious_ips}")

   # After investigation, remove it
   detector.remove_suspicious_ip(rehabilitated_ip)
   print(f"After rehabilitation: {rehabilitated_ip in detector.suspicious_ips}")

BloomFilter vs CuckooFilter
---------------------------

When to use which:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - BloomFilter
     - CuckooFilter
   * - Deletion support
     - No
     - Yes
   * - Memory efficiency
     - Better at low FP rates
     - Better at higher FP rates
   * - Lookup speed
     - Very fast
     - Very fast
   * - Use case
     - Permanent blocklists
     - Dynamic lists

Key Takeaways
-------------

1. **BloomFilter** is perfect for permanent blocklists (emails, card hashes)
2. **CuckooFilter** works better when you need deletion (IP rehabilitation)
3. **CountMinSketch** tracks repeat offenders across identities
4. **False positives are OK** — they trigger review, not automatic rejection

Exercises
---------

1. **Velocity checks**: Use HyperLogLog to count unique cards per user per hour
2. **Geographic anomalies**: Track usual login locations with a Bloom filter
3. **Time-based decay**: Create daily CuckooFilters and rotate them
4. **Cascading checks**: Use a ScalableBloomFilter for growing blocklists

Next Tutorial
-------------

Continue to :doc:`leaderboards` to learn how to build real-time trending/leaderboard systems.
