#pragma once

// Small cross-cutting types shared by multiple radarcam modules that would otherwise have no
// single natural owner. Kept minimal and dependency-free (just <array>/<string>) so both a
// lightweight consumer (extrinsics_io.hpp) and a heavy producer (clutter_filter.hpp) can include it
// without pulling unrelated machinery into each other.

#include <array>
#include <string>

namespace bev {
namespace radarcam {

// One row of the §6 clutter-filter funnel: how many detections entered this stage and how many
// survived it. Fixed-size, ordered array (not a map) -- the 7 stages are fixed and ordered by spec,
// so this shape makes misordering a compile-time-visible bug rather than a silent map-key typo.
struct StageFunnelCount {
    std::string stage_name;
    int n_entering = 0;
    int n_surviving = 0;
};

using ClutterFunnel = std::array<StageFunnelCount, 7>;

}  // namespace radarcam
}  // namespace bev
