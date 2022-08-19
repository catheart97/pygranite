#pragma once

namespace granite
{

// largest distance between to particle positions determining whether they're equal or not
constexpr float ABORT_VALUE{1e-9};

// the number of maximum supported additional volume textures
constexpr size_t MAX_ADDITIONAL_VOLUMES{8};
constexpr size_t MAX_ADDITIONAL_COMPUTE{4};

constexpr size_t NUM_CONSTANT_ADDITIONAL_COMPUTE{4};

} // namespace granite
