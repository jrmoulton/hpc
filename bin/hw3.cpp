#include <cstddef>
#include <cstdlib>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void parallel_count_sort(int a[], size_t n, size_t num_threads) {
    int *temp = (int *)malloc(n * sizeof(int));
    // set number of threads that openmp can use
    omp_set_num_threads((int)num_threads);

// signle parallel region
#pragma omp parallel
    {
        // Counting setcion
        size_t i, j;
        int count;
        // main parallel for loop. i,j,count are local to each thread.
        // static schedule
#pragma omp for schedule(static) private(i, j, count)
        for (i = 0; i < n; i++) {
            count = 0;
            for (j = 0; j < n; j++) {
                if (a[j] < a[i]) {
                    count++;
                } else if (a[j] == a[i] && j < i) {
                    count++;
                }
            }
            temp[count] = a[i];
        }

// This will happen once all of the above sections have finished
// copy section that doens't need to force a synchronize at the end
// memcpy is not thread safe so we just manually loop over the data
#pragma omp for schedule(static) nowait
        for (i = 0; i < n; i++) {
            a[i] = temp[i];
        }
    }
    free(temp);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <number of threads> <n>\n", argv[0]);
        return 1;
    }

    size_t num_threads = (size_t)atoi(argv[1]);
    size_t n = (size_t)atoi(argv[2]);

    // init array of len n with random numbers in range of 1 to n
    int *arr = (int *)malloc(n * sizeof(int));
    srand(100);
    for (size_t i = 0; i < n; i++) {
        arr[i] = (int)((size_t)rand() % n + 1);
    }

    // print original array
    printf("original: ");
    for (size_t i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    // sort array
    parallel_count_sort(arr, n, num_threads);

    // print sorted array
    printf("sorted: ");
    for (size_t i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    free(arr);
    return 0;
}
