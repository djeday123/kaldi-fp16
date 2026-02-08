#!/bin/bash
OUTPUT="snapshot/snapshot_$(date +%Y%m%d_%H%M%S).md"

mkdir -p snapshot

echo "# Kaldi-FP16 Snapshot - $(date)" > "$OUTPUT"
echo "" >> "$OUTPUT"

# Project structure
echo "## Project Structure" >> "$OUTPUT"
echo '```' >> "$OUTPUT"
tree --noreport -I 'vendor|snapshot|build|CMakeFiles' . >> "$OUTPUT" 2>/dev/null || \
find . -type f \( -name "*.go" -o -name "*.cu" -o -name "*.cpp" -o -name "*.h" -o -name "*.sh" -o -name "CMakeLists.txt" \) | grep -v vendor | grep -v snapshot | grep -v build | sort >> "$OUTPUT"
echo '```' >> "$OUTPUT"
echo "" >> "$OUTPUT"

# C++ Headers
echo "# C++ Headers" >> "$OUTPUT"
echo "" >> "$OUTPUT"
for f in $(find . -name "*.h" -o -name "*.hpp" | grep -v build | sort); do
    echo "## File: $f" >> "$OUTPUT"
    echo '```cpp' >> "$OUTPUT"
    cat "$f" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

# CUDA files
echo "# CUDA Files" >> "$OUTPUT"
echo "" >> "$OUTPUT"
for f in $(find . -name "*.cu" | grep -v build | sort); do
    echo "## File: $f" >> "$OUTPUT"
    echo '```cuda' >> "$OUTPUT"
    cat "$f" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

# C++ source files
echo "# C++ Source Files" >> "$OUTPUT"
echo "" >> "$OUTPUT"
for f in $(find . -name "*.cpp" | grep -v build | sort); do
    echo "## File: $f" >> "$OUTPUT"
    echo '```cpp' >> "$OUTPUT"
    cat "$f" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

# CMakeLists
echo "# CMake Files" >> "$OUTPUT"
echo "" >> "$OUTPUT"
for f in $(find . -name "CMakeLists.txt" | grep -v build | sort); do
    echo "## File: $f" >> "$OUTPUT"
    echo '```cmake' >> "$OUTPUT"
    cat "$f" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

# Go files
echo "# Go Files" >> "$OUTPUT"
echo "" >> "$OUTPUT"
for f in $(find . -name "*.go" | grep -v vendor | sort); do
    echo "## File: $f" >> "$OUTPUT"
    echo '```go' >> "$OUTPUT"
    cat "$f" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

# Shell scripts
echo "# Shell Scripts" >> "$OUTPUT"
echo "" >> "$OUTPUT"
for f in $(find . -name "*.sh" | grep -v snapshot | sort); do
    echo "## File: $f" >> "$OUTPUT"
    echo '```bash' >> "$OUTPUT"
    cat "$f" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

# go.mod
if [ -f "go.mod" ]; then
    echo "## File: go.mod" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    cat go.mod >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    echo "" >> "$OUTPUT"
fi

# README
if [ -f "README.md" ]; then
    echo "## File: README.md" >> "$OUTPUT"
    echo '```markdown' >> "$OUTPUT"
    cat README.md >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    echo "" >> "$OUTPUT"
fi

echo "Snapshot created: $OUTPUT"
echo "Lines: $(wc -l < "$OUTPUT")"
echo "Size: $(du -h "$OUTPUT" | cut -f1)"