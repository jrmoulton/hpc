> 1. If we try to parallelize the for i loop (the outer loop), which variables should be private and which should be shared? (5 points)

i, j, and count should be shared but the main array and the temp buffer should be shared. 

> 2. If we consider the memcpy implementation not thread-safe, how would you approach parallelizing this operation? (5 points)

It can be parallelized by just manually looping over the array and keeping it in the main parallel section but adding a new parallel for loop.
