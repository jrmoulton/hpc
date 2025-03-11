
/// this is a test
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
