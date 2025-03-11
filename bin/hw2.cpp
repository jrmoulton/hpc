#include <cstddef>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

template <typename T, typename E> struct Result {
    bool is_ok;
    union {
        T value;
        E error;
    };

    Result(const T &v) : is_ok(true), value(v) {}
    Result(const E &e) : is_ok(false), error(e) {}
    ~Result() {}
};

struct Config {
    size_t thread_count;
    size_t bin_count;
    float min_value;
    float max_value;
    size_t data_count;

    // parse config from args
    static Result<Config, const char *> from_args(int argc, char *argv[]) {
        if (argc != 6) {
            return "Wrong number of arguments";
        }

        char *end;
        Config cfg;

        cfg.thread_count = strtoul(argv[1], &end, 10);
        if (*end != '\0' || cfg.thread_count == 0) {
            return "Invalid thread count";
        }

        cfg.bin_count = strtoul(argv[2], &end, 10);
        if (*end != '\0' || cfg.bin_count == 0) {
            return "Invalid bin count";
        }

        cfg.min_value = strtof(argv[3], &end);
        if (*end != '\0') {
            return "Invalid min value";
        }

        cfg.max_value = strtof(argv[4], &end);
        if (*end != '\0') {
            return "Invalid max value";
        }

        cfg.data_count = strtoul(argv[5], &end, 10);
        if (*end != '\0' || cfg.data_count == 0) {
            return "Invalid data count";
        }

        if (cfg.min_value >= cfg.max_value) {
            return "Min value must be less than max value";
        }

        return cfg;
    }
};

// generate random data with seed 100
std::vector<float> generate_data(const Config &config) {
    std::vector<float> data;
    data.reserve(config.data_count);

    srand(100);
    const float range = config.max_value - config.min_value;
    const float max_rand = static_cast<float>(RAND_MAX);

    for (size_t i = 0; i < config.data_count; ++i) {
        float random_value = config.min_value + (static_cast<float>(rand()) / max_rand) * range;
        data.push_back(random_value);
    }

    return data;
}

// calculate bin boundaries
std::vector<float> calculate_bins(const Config &config) {
    std::vector<float> bins;
    bins.reserve(config.bin_count);

    const float bin_width =
        (config.max_value - config.min_value) / static_cast<float>(config.bin_count);

    for (size_t i = 0; i < config.bin_count; i++) {
        bins.push_back(config.min_value + static_cast<float>(i + 1) * bin_width);
    }

    return bins;
}

struct DataRange {
    size_t start;
    size_t end;
};

// calculate data ranges for threads
std::vector<DataRange> calculate_ranges(const Config &config) {
    std::vector<DataRange> ranges;
    ranges.reserve(config.thread_count);

    const size_t chunk_size = config.data_count / config.thread_count;
    const size_t remainder = config.data_count % config.thread_count;
    size_t start = 0;

    for (size_t i = 0; i < config.thread_count; i++) {
        const size_t extra = i < remainder ? 1 : 0;
        const size_t end = start + chunk_size + extra;
        DataRange range = {start, end};
        ranges.push_back(range);
        start = end;
    }

    return ranges;
}

// compute global sum in bin_counts
void global_sum(const std::vector<float> &data, const std::vector<float> &bins,
                std::vector<size_t> &bin_counts, const DataRange &range, std::mutex &mutex) {
    std::vector<size_t> local_counts(bins.size(), 0);

    // process locally to avoid lock contention
    for (size_t i = range.start; i < range.end; i++) {
        const float value = data[i];
        // find bin for value
        for (size_t bin = 0; bin < bins.size(); bin++) {
            if (value <= bins[bin]) {
                // increment local count
                local_counts[bin]++;
                break;
            }
        }
    }

    // take lock on shared bin_counts and add in locally computed values
    std::lock_guard<std::mutex> lock(mutex);
    for (size_t i = 0; i < bins.size(); i++) {
        bin_counts[i] += local_counts[i];
    }
}

// compute tree sum by creating data in thread_counts as an implicit tree that will need to be
// reduced by `reduce_tree`
void tree_sum(const std::vector<float> &data, const std::vector<float> &bins,
              std::vector<std::vector<size_t>> &thread_counts, size_t thread_id,
              const DataRange &range) {
    for (size_t i = range.start; i < range.end; i++) {
        const float value = data[i];
        // find bin for value
        for (size_t bin = 0; bin < bins.size(); bin++) {
            if (value <= bins[bin]) {
                // increment local count
                thread_counts[thread_id][bin]++;
                break;
            }
        }
    }
}

// tree reduction
std::vector<size_t> reduce_tree(std::vector<std::vector<size_t>> &thread_counts) {
    const size_t num_threads = thread_counts.size();
    const size_t bin_count = thread_counts[0].size();

    // double stride each iteration creating tree levels
    for (size_t stride = 1; stride < num_threads; stride *= 2) {
        // process pairs of thread results at current stride
        for (size_t i = 0; i < num_threads; i += stride * 2) {
            // skip if run out of pairs to combine
            if (i + stride < num_threads) {
                for (size_t bin = 0; bin < bin_count; bin++) {
                    thread_counts[i][bin] += thread_counts[i + stride][bin];
                }
            }
        }
    }

    return thread_counts[0];
}

void print_results(const std::string &method, const std::vector<float> &bins,
                   const std::vector<size_t> &counts) {
    std::cout << method << "\nbin_maxes = ";
    for (const float &bin : bins) {
        std::cout << std::fixed << std::setprecision(3) << bin << " ";
    }

    std::cout << "\nbin_counts = ";
    for (size_t count : counts) {
        std::cout << count << " ";
    }
    std::cout << "\n\n";
}

int main(int argc, char *argv[]) {
    auto config_result = Config::from_args(argc, argv);
    if (!config_result.is_ok) {
        std::cerr << "Error: " << config_result.error << "\n"
                  << "Usage: " << argv[0]
                  << " <thread_count> <bin_count> <min_value> <max_value> <data_count>\n";
        return 1;
    }
    Config config = config_result.value;

    // init data and bins
    auto data = generate_data(config);
    auto bins = calculate_bins(config);
    std::vector<DataRange> ranges = calculate_ranges(config);

    // global sum
    {
        std::vector<size_t> bin_counts(config.bin_count, 0);
        std::mutex mutex;
        std::vector<std::thread> threads;
        threads.reserve(config.thread_count);

        for (size_t i = 0; i < config.thread_count; i++) {
            threads.push_back(std::thread(global_sum, std::ref(data), std::ref(bins),
                                          std::ref(bin_counts), std::ref(ranges[i]),
                                          std::ref(mutex)));
        }

        for (std::thread &thread : threads) {
            thread.join();
        }

        print_results("Global Sum", bins, bin_counts);
    }

    // tree sum
    {
        std::vector<std::vector<size_t>> thread_counts(config.thread_count,
                                                       std::vector<size_t>(config.bin_count, 0));
        std::vector<std::thread> threads;
        threads.reserve(config.thread_count);

        for (size_t i = 0; i < config.thread_count; i++) {
            threads.push_back(std::thread(tree_sum, std::ref(data), std::ref(bins),
                                          std::ref(thread_counts), i, std::ref(ranges[i])));
        }

        for (std::thread &thread : threads) {
            thread.join();
        }

        std::vector<size_t> final_counts = reduce_tree(thread_counts);
        print_results("Tree Structured Sum", bins, final_counts);
    }

    return 0;
}
