# Serialization

Hazy supports multiple serialization formats for persisting and transferring data structures.

## Overview

| Format | Use Case | Size | Human-Readable |
|--------|----------|------|----------------|
| Binary (`to_bytes`) | Production storage | Smallest | No |
| JSON (`to_json`) | Debugging, interop | Larger | Yes |
| File (`save`/`load`) | Convenience | Same as binary | No |

## Binary Serialization

Binary format uses [bincode](https://github.com/bincode-org/bincode) for compact, efficient serialization.

### Basic Usage

```python
from hazy import BloomFilter

bf = BloomFilter(expected_items=10000)
bf.update([f"item_{i}" for i in range(5000)])

# Serialize to bytes
data = bf.to_bytes()
print(f"Size: {len(data):,} bytes")

# Deserialize
bf2 = BloomFilter.from_bytes(data)
assert "item_0" in bf2
```

### Format Details

Binary format includes:

- Magic bytes for validation
- Version number for compatibility
- Structure parameters
- Compressed data

```python
data = bf.to_bytes()

# First 4 bytes are magic number
magic = data[:4]
print(f"Magic: {magic}")  # b'HAZY'
```

### All Structures

```python
from hazy import (
    BloomFilter, CountingBloomFilter, ScalableBloomFilter,
    CuckooFilter, HyperLogLog, CountMinSketch, MinHash, TopK
)

# All structures support the same API
for cls in [BloomFilter, HyperLogLog, CountMinSketch, MinHash]:
    obj = cls(...)
    data = obj.to_bytes()
    restored = cls.from_bytes(data)
```

## JSON Serialization

JSON format is human-readable and useful for debugging or interoperability.

### Basic Usage

```python
from hazy import BloomFilter

bf = BloomFilter(expected_items=1000)
bf.add("test")

# Serialize to JSON
json_str = bf.to_json()
print(json_str[:200] + "...")

# Deserialize
bf2 = BloomFilter.from_json(json_str)
assert "test" in bf2
```

### JSON Structure

```python
import json
from hazy import HyperLogLog

hll = HyperLogLog(precision=10)
hll.update(["a", "b", "c"])

data = json.loads(hll.to_json())
print(json.dumps(data, indent=2)[:500])
```

Output shows structure parameters and register values.

### When to Use JSON

- **Debugging**: Inspect internal state
- **Interoperability**: Share with other languages
- **Version control**: Track changes in config
- **Small structures**: Acceptable size overhead

!!! warning "Size Overhead"
    JSON is typically 3-10x larger than binary format. Use binary for production.

## File I/O

Convenience methods for saving directly to files.

### Basic Usage

```python
from hazy import BloomFilter

bf = BloomFilter(expected_items=10000)
bf.update([f"item_{i}" for i in range(5000)])

# Save to file
bf.save("my_filter.hazy")

# Load from file
bf2 = BloomFilter.load("my_filter.hazy")
assert "item_0" in bf2
```

### File Format

Files use binary format with automatic compression detection:

```python
import os

bf = BloomFilter(expected_items=100000)
bf.update([f"item_{i}" for i in range(50000)])

bf.save("filter.hazy")
print(f"File size: {os.path.getsize('filter.hazy'):,} bytes")
```

### Path Handling

```python
from pathlib import Path

# String paths
bf.save("data/filters/my_filter.hazy")

# Path objects
bf.save(Path("data") / "filters" / "my_filter.hazy")

# Absolute paths
bf.save("/var/data/filter.hazy")
```

## Size Comparison

```python
from hazy import BloomFilter
import json

bf = BloomFilter(expected_items=100000)
bf.update([f"item_{i}" for i in range(50000)])

binary_size = len(bf.to_bytes())
json_size = len(bf.to_json().encode())

print(f"Binary: {binary_size:,} bytes")
print(f"JSON: {json_size:,} bytes")
print(f"Ratio: {json_size / binary_size:.1f}x")
```

Typical output:
```
Binary: 12,456 bytes
JSON: 89,234 bytes
Ratio: 7.2x
```

## Versioning and Compatibility

### Version Checking

```python
data = bf.to_bytes()

# Version is embedded in the format
# Incompatible versions raise an error
try:
    bf2 = BloomFilter.from_bytes(data)
except ValueError as e:
    print(f"Incompatible version: {e}")
```

### Forward Compatibility

Hazy maintains forward compatibility within major versions:

- Data from v0.1.0 can be read by v0.1.x
- Major version changes may require migration

## Error Handling

### Corrupted Data

```python
from hazy import BloomFilter

# Corrupted data raises ValueError
try:
    bf = BloomFilter.from_bytes(b"corrupted data")
except ValueError as e:
    print(f"Failed to deserialize: {e}")
```

### File Not Found

```python
from hazy import BloomFilter

try:
    bf = BloomFilter.load("nonexistent.hazy")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

### Type Mismatch

```python
from hazy import BloomFilter, HyperLogLog

bf = BloomFilter(expected_items=1000)
data = bf.to_bytes()

# Wrong type raises ValueError
try:
    hll = HyperLogLog.from_bytes(data)
except ValueError as e:
    print(f"Type mismatch: {e}")
```

## Use Cases

### 1. Periodic Snapshots

```python
from hazy import BloomFilter
import time

bf = BloomFilter(expected_items=1_000_000)

def snapshot():
    timestamp = int(time.time())
    bf.save(f"snapshots/filter_{timestamp}.hazy")

# Save hourly snapshots
while True:
    process_events(bf)
    snapshot()
    time.sleep(3600)
```

### 2. Distributed Systems

```python
from hazy import HyperLogLog
import redis

r = redis.Redis()

def save_to_redis(key, hll):
    r.set(key, hll.to_bytes())

def load_from_redis(key):
    data = r.get(key)
    return HyperLogLog.from_bytes(data) if data else None

# Worker saves local HLL
local_hll = HyperLogLog(precision=14)
# ... add items ...
save_to_redis(f"hll:worker:{worker_id}", local_hll)

# Aggregator merges all
merged = HyperLogLog(precision=14)
for key in r.scan_iter("hll:worker:*"):
    worker_hll = load_from_redis(key)
    merged.merge(worker_hll)
```

### 3. API Response Caching

```python
from hazy import CountMinSketch
import base64

def serialize_for_api(cms):
    """Encode for JSON API response."""
    return base64.b64encode(cms.to_bytes()).decode()

def deserialize_from_api(encoded):
    """Decode from JSON API response."""
    return CountMinSketch.from_bytes(base64.b64decode(encoded))

# API endpoint
@app.get("/sketch")
def get_sketch():
    return {"data": serialize_for_api(global_sketch)}
```

## Best Practices

1. **Use binary for production** - Smaller size, faster serialization

2. **Use JSON for debugging** - Inspect internal state when troubleshooting

3. **Include metadata** - Store creation time, parameters, etc. alongside

4. **Handle errors gracefully** - Always wrap deserialization in try/except

5. **Version your snapshots** - Include version info in filenames or metadata

```python
import json
from datetime import datetime

def save_with_metadata(bf, path):
    """Save filter with metadata."""
    metadata = {
        "created": datetime.now().isoformat(),
        "expected_items": bf.num_bits // 10,  # approximate
        "version": "0.1.0"
    }

    with open(f"{path}.meta.json", "w") as f:
        json.dump(metadata, f)

    bf.save(path)
```
