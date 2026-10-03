#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

struct Score {
    int order;
    int32_t idx;
    float val;
    bool operator>(const Score& other) const { return val > other.val; }
};

struct Candidate {
    int32_t idx;
    int32_t key;
    float val;
};

static std::vector<Score> replay(const Candidate* input, int width) {
    std::vector<Candidate> stream(input, input + width);
    std::sort(stream.begin(), stream.end(), [](const Candidate& a, const Candidate& b) {
        return a.key > b.key;
    });
    constexpr int k = 20;
    std::vector<Score> heap(k, {2147483647, -1, 1e-8f});
    std::make_heap(heap.begin(), heap.end(), std::greater<Score>());
    std::pop_heap(heap.begin(), heap.end(), std::greater<Score>());
    int nset = 0;
    for (const auto& c : stream) {
        if (c.idx < 0 || !(c.val > heap.front().val)) continue;
        std::pop_heap(heap.begin(), heap.end(), std::greater<Score>());
        heap.back() = {nset++, c.idx, c.val};
        std::push_heap(heap.begin(), heap.end(), std::greater<Score>());
    }
    std::sort(heap.begin(), heap.end(), std::greater<Score>());
    return heap;
}

extern "C" void sdtn_value_sort(int nrows, int width, int32_t* orders,
                                  int32_t* indices, float* values) {
    std::vector<Score> row(width);
    for (int r = 0; r < nrows; ++r) {
        for (int j = 0; j < width; ++j) {
            const int p = r * width + j;
            row[j] = {orders[p], indices[p], values[p]};
        }
        std::sort(row.begin(), row.end(), std::greater<Score>());
        for (int j = 0; j < width; ++j) {
            const int p = r * width + j;
            orders[p] = row[j].order;
            indices[p] = row[j].idx;
            values[p] = row[j].val;
        }
    }
}

extern "C" void sdtn_replay_selected(int nrows, int width, const int32_t* keys,
                                      int32_t* indices, float* values) {
    std::vector<Candidate> input(width);
    for (int r = 0; r < nrows; ++r) {
        for (int j = 0; j < width; ++j) {
            const int p = r * width + j;
            input[j] = {indices[p], keys[p], values[p]};
        }
        auto heap = replay(input.data(), width);
        for (int j = 0; j < 20; ++j) {
            const int p = r * width + j;
            indices[p] = heap[j].idx;
            values[p] = heap[j].val;
        }
    }
}
