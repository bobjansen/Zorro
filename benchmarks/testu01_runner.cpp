#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <variant>

extern "C" {
#include <testu01/bbattery.h>
#include <testu01/unif01.h>
}

#include "benchmarks/raw_stream_sources.hpp"

namespace {

constexpr std::uint64_t kDefaultSeed = 0x123456789abcdef0ULL;

using StreamVariant =
    std::variant<zorro_bench::ScalarEmitter, zorro_bench::X2Emitter, zorro_bench::X4Emitter>;

std::unique_ptr<StreamVariant> g_stream;
std::uint64_t g_word = 0;
bool g_have_low_half = false;

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <smallcrush|crush|bigcrush> <scalar|x2|x4> [seed]\n";
}

auto next_word() -> std::uint64_t {
    return std::visit([](auto& stream) { return stream(); }, *g_stream);
}

auto next_bits32() -> unsigned int {
    if (!g_have_low_half) {
        g_word = next_word();
        g_have_low_half = true;
        return static_cast<unsigned int>(g_word);
    }

    g_have_low_half = false;
    return static_cast<unsigned int>(g_word >> 32);
}

void write_state() {
    std::cout << "stream state managed internally" << '\n';
}

auto make_stream(const std::string& mode, std::uint64_t seed) -> StreamVariant {
    if (mode == "scalar")
        return zorro_bench::ScalarEmitter(seed);
    if (mode == "x2")
        return zorro_bench::X2Emitter(seed);
    return zorro_bench::X4Emitter(seed);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        usage(argv[0]);
        return 1;
    }

    const std::string battery = argv[1];
    const std::string mode = argv[2];
    if (battery != "smallcrush" && battery != "crush" && battery != "bigcrush") {
        usage(argv[0]);
        return 1;
    }
    if (mode != "scalar" && mode != "x2" && mode != "x4") {
        usage(argv[0]);
        return 1;
    }

    const std::uint64_t seed =
        argc == 4 ? std::strtoull(argv[3], nullptr, 0) : kDefaultSeed;
    const std::string name = "zorro_xoshiro256pp_" + mode;

    g_stream = std::make_unique<StreamVariant>(make_stream(mode, seed));
    g_have_low_half = false;

    unif01_Gen* gen = unif01_CreateExternGenBits(
        const_cast<char*>(name.c_str()), &next_bits32);
    gen->Write = [](void*) { write_state(); };

    if (battery == "smallcrush") {
        bbattery_SmallCrush(gen);
    } else if (battery == "crush") {
        bbattery_Crush(gen);
    } else {
        bbattery_BigCrush(gen);
    }

    unif01_DeleteExternGenBits(gen);
    g_stream.reset();
    return 0;
}
