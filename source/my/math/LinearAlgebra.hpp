#pragma once

#ifdef MY_USE_CUDA_TYPES
#include <cuda.h>
#include <vector_types.h>
#endif

#include <cmath>
#include <ostream>

namespace my
{

namespace math
{

/////////////
/// DATATYPES
/////////////

#ifdef MY_USE_CUDA_TYPES
using Vec4 = float4;
#else
struct Vec4
{
    // data
public:
    float x, y, z, w;
};
#endif

#ifdef MY_USE_CUDA_TYPES
using Vec3 = float3;
#else
struct Vec3
{
    // data
public:
    float x, y, z;
};
#endif

#ifdef MY_USE_CUDA_TYPES
using Vec2 = float2;
#else
struct Vec2
{
    // data
public:
    float x, y;
};
#endif

using Mat2x2 = Vec4;

struct Mat3x3
{
    // data
public:
    Vec3 data[3];

    // operators
public:
#ifdef __NVCC__
    __device__ __host__ __forceinline__
#else
    inline
#endif
        float &
        operator()(size_t i, size_t j)
    {
        return reinterpret_cast<float *>(&data[i])[j];
    }
};

struct Mat4x4
{
    // data
public:
    Vec4 data[4];

    // operators
public:
#ifdef __NVCC__
    __device__ __host__ __forceinline__
#else
    inline
#endif
        float &
        operator()(size_t i, size_t j)
    {
        return reinterpret_cast<float *>(&data[i])[j];
    }
};

using Quat = Vec4;

//////////////
/// OPERATIONS
//////////////

/// TRANSPOSITION
#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat2x2
    transpose(Mat2x2 & m)
{
    return {m.x, m.z, m.y, m.w};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat3x3
    transpose(Mat3x3 & m)
{
    Mat3x3 res;
    for (size_t i{0}; i < 3; ++i)
    {
        for (size_t j{0}; j < 3; ++j)
        {
            res(i, j) = m(j, i);
        }
    }
    return res;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat4x4
    transpose(Mat4x4 & m)
{
    Mat4x4 res;
    for (size_t i{0}; i < 4; ++i)
    {
        for (size_t j{0}; j < 4; ++j)
        {
            res(i, j) = m(j, i);
        }
    }
    return res;
}

/// ADDITION

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec4
    operator+(const Vec4 & a, const Vec4 & b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec3
    operator+(const Vec3 & a, const Vec3 & b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec2
    operator+(const Vec2 & a, const Vec2 & b)
{
    return {a.x + b.x, a.y + b.y};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat3x3
    operator+(const Mat3x3 & a, const Mat3x3 & b)
{
    Mat3x3 res;
    for (size_t i{0}; i < 3; ++i)
    {
        res.data[i] = a.data[i] + b.data[i];
    }
    return res;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat4x4
    operator+(const Mat4x4 & a, const Mat4x4 & b)
{
    Mat4x4 res;
    for (size_t i{0}; i < 4; ++i)
    {
        res.data[i] = a.data[i] + b.data[i];
    }
    return res;
}

/// SUBSTRACTION

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec4
    operator-(const Vec4 & a, const Vec4 & b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec3
    operator-(const Vec3 & a, const Vec3 & b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec2
    operator-(const Vec2 & a, const Vec2 & b)
{
    return {a.x - b.x, a.y - b.y};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat3x3
    operator-(const Mat3x3 & a, const Mat3x3 & b)
{
    Mat3x3 res;
    for (size_t i{0}; i < 3; ++i)
    {
        res.data[i] = a.data[i] - b.data[i];
    }
    return res;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat4x4
    operator-(const Mat4x4 & a, const Mat4x4 & b)
{
    Mat4x4 res;
    for (size_t i{0}; i < 4; ++i)
    {
        res.data[i] = a.data[i] - b.data[i];
    }
    return res;
}

/// SCALAR MULTIPLICATION

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec4
    operator*(const Vec4 & v, float s)
{
    return {s * v.x, s * v.y, s * v.z, s * v.w};
};

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec4
    operator*(float s, const Vec4 & v)
{
    return {s * v.x, s * v.y, s * v.z, s * v.w};
};

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec3
    operator*(const Vec3 & v, float s)
{
    return {s * v.x, s * v.y, s * v.z};
};

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec3
    operator*(float s, const Vec3 & v)
{
    return {s * v.x, s * v.y, s * v.z};
};

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec2
    operator*(const Vec2 & v, float s)
{
    return {s * v.x, s * v.y};
};

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec2
    operator*(float s, const Vec2 & v)
{
    return {s * v.x, s * v.y};
};

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec4
    operator/(const Vec4 & v, float s)
{
    return {v.x / s, v.y / s, v.z / s, v.w / s};
};

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec3
    operator/(const Vec3 & v, float s)
{
    return {v.x / s, v.y / s, v.z / s};
};

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec2
    operator/(const Vec2 & v, float s)
{
    return {v.x / s, v.y / s};
};

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat3x3
    operator*(const Mat3x3 & a, float b)
{
    Mat3x3 res;
    for (size_t i{0}; i < 3; ++i)
    {
        res.data[i] = a.data[i] * b;
    }
    return res;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat3x3
    operator*(float b, const Mat3x3 & a)
{
    Mat3x3 res;
    for (size_t i{0}; i < 3; ++i)
    {
        res.data[i] = a.data[i] * b;
    }
    return res;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat4x4
    operator*(const Mat4x4 & a, float b)
{
    Mat4x4 res;
    for (size_t i{0}; i < 4; ++i)
    {
        res.data[i] = a.data[i] * b;
    }
    return res;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat4x4
    operator*(float b, const Mat4x4 & a)
{
    Mat4x4 res;
    for (size_t i{0}; i < 4; ++i)
    {
        res.data[i] = a.data[i] * b;
    }
    return res;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat3x3
    operator/(const Mat3x3 & a, float b)
{
    Mat3x3 res;
    for (size_t i{0}; i < 3; ++i)
    {
        res.data[i] = a.data[i] / b;
    }
    return res;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat4x4
    operator/(const Mat4x4 & a, float b)
{
    Mat4x4 res;
    for (size_t i{0}; i < 4; ++i)
    {
        res.data[i] = a.data[i] / b;
    }
    return res;
}

/// EQUALITY

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr bool
    operator==(const Vec4 & a, const Vec4 & b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr bool
    operator==(const Vec3 & a, const Vec3 & b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr bool
    operator==(const Vec2 & a, const Vec2 & b)
{
    return a.x == b.x && a.y == b.y;
}

/// SPECIAL VECTOR FUNCTIONS

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr float
    dot(const Vec4 & v, const Vec4 & o)
{
    return v.x * o.x + v.y * o.y + v.z * o.z + v.w * o.w;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr float
    dot(const Vec3 & v, const Vec3 & o)
{
    return v.x * o.x + v.y * o.y + v.z * o.z;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr float
    dot(const Vec2 & v, const Vec2 & o)
{
    return v.x * o.x + v.y * o.y;
}

template <typename VEC_T>
#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr float
    length_squared(const VEC_T & v)
{
    return dot(v, v);
}

template <typename VEC_T>
#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr float
    length(const VEC_T & v)
{
    return sqrt(length_squared(v));
}

template <typename VEC_T>
#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr VEC_T
    normalized(const VEC_T & v)
{
    return v * rsqrt(length_squared(v));
}

template <typename VEC_T>
#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr float
    angle(const VEC_T & a, const VEC_T & b)
{
    return acos(dot(normalized(a), normalized(b)));
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr Vec3
    cross(const Vec3 & a, const Vec3 & b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    float
    lerp(const float a, const float b, const float t)
{
    return a + t * (b - a);
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    my::math::Vec3
    lerp(const my::math::Vec3 & a, const my::math::Vec3 & b, float t)
{
    return my::math::Vec3{lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t)};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    my::math::Vec2
    lerp(const my::math::Vec2 & a, const my::math::Vec2 & b, float t)
{
    return my::math::Vec2{lerp(a.x, b.x, t), lerp(a.y, b.y, t)};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    float
    unit_interval(float arg, const float & a, const float & b)
{
    return (arg - b) / (a - b);
}

/// MULTIPLICATION

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat2x2
    operator*(Mat2x2 & m1, Mat2x2 & m2)
{
    return {m1.x * m2.x + m1.y * m2.z, //
            m1.x * m2.y + m1.y * m2.w, //
            m1.z * m2.x + m1.w * m2.z, //
            m1.z * m2.y + m1.w * m2.w};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat3x3
    operator*(Mat3x3 & m1, Mat3x3 & m2)
{
    Mat3x3 res, m2t{transpose(m2)};
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) res(i, j) = dot(m1.data[i], m2t.data[j]);
    return res;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Mat4x4
    operator*(Mat4x4 & m1, Mat4x4 & m2)
{
    Mat4x4 res, m2t{transpose(m2)};
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) res(i, j) = dot(m1.data[i], m2t.data[j]);
    return res;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Vec2
    operator*(Mat2x2 & m1, Vec2 & m2)
{
    return {dot(Vec2{m1.x, m1.y}, m2), dot(Vec2{m1.z, m1.w}, m2)};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Vec3
    operator*(Mat3x3 & m1, Vec3 & m2)
{
    return {dot(m1.data[0], m2), dot(m1.data[1], m2), dot(m1.data[2], m2)};
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Vec4
    operator*(Mat4x4 & m1, Vec4 & m2)
{
    return {dot(m1.data[0], m2), dot(m1.data[1], m2), dot(m1.data[2], m2), dot(m1.data[3], m2)};
}

/// OUT STREAM

inline std::ostream & operator<<(std::ostream & os, const Vec4 & v)
{
    os << "Vec4 " << v.x << " " << v.y << " " << v.z << " " << v.w;
    return os;
}

inline std::ostream & operator<<(std::ostream & os, const Vec3 & v)
{
    os << "Vec3 " << v.x << " " << v.y << " " << v.z;
    return os;
}

inline std::ostream & operator<<(std::ostream & os, const Vec2 & v)
{
    os << "Vec2 " << v.x << " " << v.y;
    return os;
}

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    Quat
    RollPitchYaw(float yaw, float pitch, float roll)
{
    float cy = cosf(yaw * 0.5f);
    float sy = sinf(yaw * 0.5f);
    float cp = cosf(pitch * 0.5f);
    float sp = sinf(pitch * 0.5f);
    float cr = cosf(roll * 0.5f);
    float sr = sinf(roll * 0.5f);

    Quat q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

} // namespace math

} // namespace my
