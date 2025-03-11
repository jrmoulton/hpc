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

# Run a binary with MPI and optional arguments
# Usage: just mpi-run <binary> <num_processes> [args...]
mpi-run binary procs *args : (build binary)
    mpiexec -n {{procs}} ./target/{{binary}} {{args}}

# Run a binary with MPI on specific hosts
# Usage: just mpi-host-run <binary> <num_processes> <hostfile> [args...]
mpi-host-run binary procs hostfile *args : (build binary)
    mpiexec -n {{procs}} --hostfile {{hostfile}} ./target/{{binary}} {{args}}

# Compile and run an MPI program in one step (development convenience)
mpi binary procs *args:
    zig build {{binary}} && mpiexec -n {{procs}} ./target/{{binary}} {{args}}


clean:
    rm -rf ".cache" ".zig-cache" "target/" 
