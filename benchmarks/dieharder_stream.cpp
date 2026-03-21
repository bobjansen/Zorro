#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "zorro/zorro.hpp"

namespace {

constexpr std::uint64_t kDefaultSeed = 0x123456789abcdef0ULL;

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <scalar|x2|x4> [seed] [count]\n"
              << "  count is the number of 64-bit words to emit; omit for unlimited output\n";
}

template <typename EmitFn>
auto stream_words(std::uint64_t count, EmitFn&& emit_next) -> int {
    if (count == 0) {
        while (true) {
            const std::uint64_t word = emit_next();
            std::cout.write(reinterpret_cast<const char*>(&word), sizeof(word));
            if (!std::cout)
                return std::cout.eof() ? 0 : 1;
        }
    }

    for (std::uint64_t i = 0; i < count; ++i) {
        const std::uint64_t word = emit_next();
        std::cout.write(reinterpret_cast<const char*>(&word), sizeof(word));
        if (!std::cout)
            return std::cout.eof() ? 0 : 1;
    }
    return 0;
}

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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        usage(argv[0]);
        return 1;
    }

    const std::string mode = argv[1];
    const std::uint64_t seed =
        argc >= 3 ? std::strtoull(argv[2], nullptr, 0) : kDefaultSeed;
    const std::uint64_t count =
        argc >= 4 ? std::strtoull(argv[3], nullptr, 0) : 0;

    std::cout.rdbuf()->pubsetbuf(nullptr, 0);

    if (mode == "scalar") {
        zorro::Xoshiro256pp rng(seed);
        return stream_words(count, [&] { return rng(); });
    }
    if (mode == "x2") {
        X2Emitter emitter(seed);
        return stream_words(count, emitter);
    }
    if (mode == "x4") {
        X4Emitter emitter(seed);
        return stream_words(count, emitter);
    }

    usage(argv[0]);
    return 1;
}
