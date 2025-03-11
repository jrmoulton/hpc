#include <cstddef>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>

struct Config {
    size_t bin_count;
    float min_value;
    float max_value;
    size_t data_count;

    // parse config from args
    static bool from_args(int argc, char *argv[], Config &cfg) {
        if (argc != 5) {
            return false;
        }

        char *end;

        cfg.bin_count = strtoul(argv[1], &end, 10);
        if (*end != '\0' || cfg.bin_count <= 0) {
            return false;
        }

        cfg.min_value = strtof(argv[2], &end);
        if (*end != '\0') {
            return false;
        }

        cfg.max_value = strtof(argv[3], &end);
        if (*end != '\0') {
            return false;
        }

        cfg.data_count = strtoul(argv[4], &end, 10);
        if (*end != '\0' || cfg.data_count <= 0) {
            return false;
        }

        if (cfg.min_value >= cfg.max_value) {
            return false;
        }

        return true;
    }
};

// generate random data with seed 100
std::vector<float> generate_data(const Config &config) {
    std::vector<float> data(config.data_count);

    srand(100);
    const float range = config.max_value - config.min_value;
    const float max_rand = static_cast<float>(RAND_MAX);

    for (size_t i = 0; i < config.data_count; ++i) {
        float random_value = config.min_value + (static_cast<float>(rand()) / max_rand) * range;
        data[i] = random_value;
    }

    return data;
}

// calculate bin boundaries
std::vector<float> calculate_bins(const Config &config) {
    std::vector<float> bins(config.bin_count);

    const float bin_width =
        (config.max_value - config.min_value) / static_cast<float>(config.bin_count);

    for (size_t i = 0; i < config.bin_count; i++) {
        bins[i] = config.min_value + static_cast<float>(i + 1) * bin_width;
    }

    return bins;
}

// compute local histogram counts
void compute_local_histogram(const std::vector<float> &local_data, const std::vector<float> &bins,
                             std::vector<int> &local_counts) {
    // initialize local counts
    for (size_t i = 0; i < local_counts.size(); i++) {
        local_counts[i] = 0;
    }

    // count data values into bins
    for (size_t i = 0; i < local_data.size(); i++) {
        const float value = local_data[i];
        // find bin for value
        for (size_t bin = 0; bin < bins.size(); bin++) {
            if (value <= bins[bin]) {
                // increment local count
                local_counts[bin]++;
                break;
            }
        }
    }
}

// print the final results
void print_results(const std::vector<float> &bins, const std::vector<int> &counts) {
    std::cout << "bin_maxes: ";
    for (const float &bin : bins) {
        std::cout << std::fixed << std::setprecision(3) << bin << " ";
    }
    std::cout << std::endl;

    std::cout << "bin_counts: ";
    for (int count : counts) {
        std::cout << count << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    // init MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Config config;
    std::vector<float> data;
    std::vector<float> bins;
    std::vector<int> global_counts;

    // process 0 reads inputs and initializes data
    if (rank == 0) {
        if (!Config::from_args(argc, argv, config)) {
            std::cerr << "Error: Invalid arguments\n"
                      << "Usage: " << argv[0]
                      << " <bin_count> <min_meas> <max_meas> <data_count>\n";
            return 1;
        }

        // generate random data
        data = generate_data(config);

        // calculate bin boundaries
        bins = calculate_bins(config);

        // init global histogram counts
        global_counts.resize(config.bin_count, 0);
    }

    // broadcast config to all processes
    MPI_Bcast(&config.bin_count, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&config.min_value, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&config.max_value, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&config.data_count, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // all processes init their bin boundries
    if (rank != 0) {
        bins = calculate_bins(config);
    }

    // calculate local data sizes and displacements for scatter
    int *send_counts = new int[size];
    int *displs = new int[size];

    int base_count = static_cast<int>(config.data_count / (size_t)size);
    int remainder = static_cast<int>(config.data_count % (size_t)size);

    int total_sent = 0;
    for (int i = 0; i < size; i++) {
        send_counts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i] = total_sent;
        total_sent += send_counts[i];
    }

    // allocte memory for local data to receive from MPI_Scatterv
    std::vector<float> local_data(static_cast<size_t>(send_counts[rank]));

    // distribute data using Scatterv (using this because I am splitting up remainders)
    MPI_Scatterv(rank == 0 ? data.data() : nullptr, send_counts, displs, MPI_FLOAT,
                 local_data.data(), send_counts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    // compute local histogram
    std::vector<int> local_counts(config.bin_count, 0);
    compute_local_histogram(local_data, bins, local_counts);

    if (rank == 0) {
        global_counts.resize(config.bin_count, 0);
    }

    // reduce local histograms into the global histogram var
    MPI_Reduce(local_counts.data(), global_counts.data(), (int)config.bin_count, MPI_INT, MPI_SUM,
               0, MPI_COMM_WORLD);

    // print results from proc o
    if (rank == 0) {
        print_results(bins, global_counts);
    }

    MPI_Finalize();
    return 0;
}
