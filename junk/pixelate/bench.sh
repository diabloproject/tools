#!/bin/bash
# Benchmark script for Pixelate Data Node

cd "$(dirname "$0")"

echo "=== Pixelate Data Node Benchmark Suite ==="
echo ""

# Check if data node is running
if ! curl -s http://127.0.0.1:50051 > /dev/null 2>&1; then
    echo "⚠️  Data node doesn't appear to be running on port 50051"
    echo "    Start it with: cd data/dnd/1 && ../../../target/release/data-node"
    echo ""
fi

# Parse arguments
TEST_TYPE="${1:-all}"
DURATION="${2:-10}"
CONCURRENCY="${3:-10}"

echo "Running benchmark..."
echo "  Test type: $TEST_TYPE"
echo "  Duration: ${DURATION}s"
echo "  Concurrency: $CONCURRENCY"
echo ""

./target/release/pixelate-bench "$TEST_TYPE" "$DURATION" "$CONCURRENCY"

echo ""
echo "=== Benchmark Complete ==="
echo ""
echo "Usage: $0 [test_type] [duration_secs] [concurrency]"
echo "  test_type: push, read, snapshot, or all (default: all)"
echo "  duration_secs: test duration in seconds (default: 10)"
echo "  concurrency: number of concurrent workers (default: 10)"
echo ""
echo "Examples:"
echo "  $0 push 30 50       # Push pixels for 30s with 50 workers"
echo "  $0 read 10 100      # Read pixels for 10s with 100 workers"
echo "  $0 snapshot 5       # Snapshot for 5s (always sequential)"
echo "  $0 all 15 20        # Run all tests for 15s with 20 workers"
