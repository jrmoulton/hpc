Block Size 1: 8×8 (64 threads per block) 8×8 blocks. 25% GPU usage. 512 threads per SM instead
of 2048. Too many small blocks. Slowest option.

Block Size 2: 16×16 (256 threads per block) 16×16 blocks. 100% GPU usage. Full 2048 threads per SM.
Perfect alignment. Optimal performance.

Block Size 3: 32×32 (1024 threads per block) 32×32 blocks. 100% usage. Only 2 blocks per SM. Blocks
probably too large.
