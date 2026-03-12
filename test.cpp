#include <array>
#include <cstdint>

namespace detail {
inline constexpr std::uint64_t stream_offsets[4] = {
    0x0000000000000000ULL,
    0x9e3779b97f4a7c15ULL,
    0x6c62272e07bb0142ULL,
    0xd2a98b26625eee7bULL,
};
}  // namespace detail

inline auto splitmix64(std::uint64_t& state) noexcept -> std::uint64_t {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Xoshiro256pp_x4_portable {
    std::uint64_t s[4][4];  // s[word][lane]

    explicit Xoshiro256pp_x4_portable(std::uint64_t seed) noexcept {
        for (int lane = 0; lane < 4; ++lane) {
            std::uint64_t lseed = seed ^ detail::stream_offsets[lane];
            s[0][lane] = splitmix64(lseed);
            s[1][lane] = splitmix64(lseed);
            s[2][lane] = splitmix64(lseed);
            s[3][lane] = splitmix64(lseed);
        }
    }

    [[nodiscard]] std::array<std::uint64_t, 4> operator()() noexcept {
        std::uint64_t result[4];
        for (int i = 0; i < 4; ++i)
            result[i] = std::rotl(s[0][i] + s[3][i], 23) + s[0][i];

        std::uint64_t t[4];
        for (int i = 0; i < 4; ++i)
            t[i] = s[1][i] << 17;

        for (int i = 0; i < 4; ++i)
            s[2][i] ^= s[0][i];
        for (int i = 0; i < 4; ++i)
            s[3][i] ^= s[1][i];
        for (int i = 0; i < 4; ++i)
            s[1][i] ^= s[2][i];
        for (int i = 0; i < 4; ++i)
            s[0][i] ^= s[3][i];
        for (int i = 0; i < 4; ++i)
            s[2][i] ^= t[i];
        for (int i = 0; i < 4; ++i)
            s[3][i] = std::rotl(s[3][i], 45);

        return {result[0], result[1], result[2], result[3]};
    }
};

int main(int argc, char *argv[]) {
    Xoshiro256pp_x4_portable rng{static_cast<uint64_t>(argc)};
    auto res = rng();
    for (int i{}; i < 255; ++i) res = rng();
    return res[0] ^ res[1] ^ res[2] ^ res[3];
}

