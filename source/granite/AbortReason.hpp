#pragma once

#include <cstdint>

namespace granite
{

enum class AbortReason : int32_t
{
    Time = 0, // equals to alive during simulation
    Length = 1,
    Domain = 2,
    Topography = 3,
    Wind = 4
};

}