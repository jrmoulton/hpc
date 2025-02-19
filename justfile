set quiet

# List all available binaries
list:
    @ls bin/*.cpp | sd "bin/" "" | sd ".cpp" ""

# Build all binaries
[no-quiet]
build-all:
    zig build all

# Build a specific binary
build binary:
    zig build {{binary}}


# Run a specific binary with optional arguments
run binary *args : (build binary)
    ./target/{{binary}} {{args}}
