#pragma once

#include <array>
#include <cstdint>

#include "zorro/zorro.hpp"

namespace zorro_bench {

struct ScalarEmitter {
    explicit ScalarEmitter(std::uint64_t seed) : rng(seed) {}

    auto operator()() -> std::uint64_t { return rng(); }

    zorro::Xoshiro256pp rng;
};

struct X2Emitter {
    explicit X2Emitter(std::uint64_t seed) : rng(seed) {}

    auto operator()() -> std::uint64_t {
        if (index == buffer.size()) {
            buffer = rng();
            index = 0;
        }
        return buffer[index++];
    }

    zorro::Xoshiro256pp_x2 rng;
    std::array<std::uint64_t, 2> buffer = {};
    std::size_t index = 2;
};

struct X4Emitter {
    explicit X4Emitter(std::uint64_t seed) : rng(seed) {}

    auto operator()() -> std::uint64_t {
        if (index == buffer.size()) {
            buffer = rng();
            index = 0;
        }
        return buffer[index++];
    }

    zorro::Xoshiro256pp_x4_portable rng;
    std::array<std::uint64_t, 4> buffer = {};
    std::size_t index = 4;
};

}  // namespace zorro_bench
