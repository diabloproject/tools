#!/bin/bash
# Demo script for ASS to LLMSUB converter

echo "=== ASS to LLMSUB Converter Demo ==="
echo

echo "Building the converter..."
cargo build --release
echo

echo "Converting sample ASS file with default settings..."
cargo run --release -- -i examples/sample.ass -o examples/sample.llmsub
echo

echo "Generated LLMSUB content:"
echo "========================"
cat examples/sample.llmsub
echo
echo "========================"
echo

echo "Converting with different options (no grouping, 5s break threshold)..."
cargo run --release -- -i examples/sample.ass -o examples/sample_alt.llmsub --no-grouping --break-threshold 5.0
echo

echo "Alternative version (first 30 lines):"
echo "======================================"
head -30 examples/sample_alt.llmsub
echo "======================================"
echo

echo "Demo complete! Check the examples/ directory for output files."
echo "Use 'cargo run -- --help' to see all available options."