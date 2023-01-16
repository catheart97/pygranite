#pragma once

namespace granite
{

/**
 * @brief Template for a X, Y, Z dimension type (Used for grid, domain, offset and scale)
 *
 * @author Ronja Schnur (ronjaschnur@uni-mainz.de)
 */
template <typename value_t> struct Info
{
    value_t X, Y, Z;
};

using GridInfo = Info<size_t>;
using DomainInfo = Info<float>;
using OffsetInfo = Info<float>;
using ScaleInfo = Info<float>;

} // namespace granite