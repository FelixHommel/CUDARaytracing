#ifndef CRT_SRC_VEC_3_HPP
#define CRT_SRC_VEC_3_HPP

#include <cuda_runtime_api.h>
#include <fmt/base.h>

#include <cmath>
#include <iostream>

/// \brief CUDA compatible mathematical three-dimensional vector class.
///
/// \author Felix Hommel
/// \date 6/5/2026
class Vec3
{
public:
    __host__ __device__ constexpr explicit Vec3() {} // NOLINT
    __host__ __device__ constexpr explicit Vec3(float value)
    {
        m_data[0] = value;
        m_data[1] = value;
        m_data[2] = value;
    }
    __host__ __device__ constexpr Vec3(float x, float y, float z)
    {
        m_data[0] = x;
        m_data[1] = y;
        m_data[2] = z;
    }

    __host__ __device__ constexpr inline float x() const noexcept { return m_data[0]; }
    __host__ __device__ constexpr inline float y() const noexcept { return m_data[1]; }
    __host__ __device__ constexpr inline float z() const noexcept { return m_data[2]; }
    __host__ __device__ constexpr inline float r() const noexcept { return m_data[0]; }
    __host__ __device__ constexpr inline float g() const noexcept { return m_data[1]; }
    __host__ __device__ constexpr inline float b() const noexcept { return m_data[2]; }

    __host__ __device__ constexpr inline const Vec3& operator+() const { return *this; }
    __host__ __device__ constexpr inline Vec3 operator-() const { return { -m_data[0], -m_data[1], -m_data[2] }; };

    __host__ __device__ constexpr inline float operator[](unsigned int i) const { return m_data[i]; } // NOLINT
    __host__ __device__ constexpr inline float& operator[](unsigned int i) { return m_data[i]; }      // NOLINT

    __host__ __device__ constexpr inline Vec3& operator+=(const Vec3& other);
    __host__ __device__ constexpr inline Vec3& operator-=(const Vec3& other);
    __host__ __device__ constexpr inline Vec3& operator*=(const Vec3& other);
    __host__ __device__ constexpr inline Vec3& operator/=(const Vec3& other);
    __host__ __device__ constexpr inline Vec3& operator*=(float scalar);
    __host__ __device__ constexpr inline Vec3& operator/=(float scalar);

    __host__ __device__ constexpr inline float lengthSquared() const
    {
        return (m_data[0] * m_data[0]) + (m_data[1] * m_data[1]) + (m_data[2] * m_data[2]);
    }
    __host__ __device__ constexpr inline float length() const { return std::sqrt(lengthSquared()); }
    __host__ __device__ constexpr inline void normalize();

    friend inline std::istream& operator>>(std::istream& in, Vec3& v);
    friend inline std::ostream& operator<<(std::ostream& out, const Vec3& v);

private:
    float m_data[3]; // NOLINT
};

inline std::istream& operator>>(std::istream& in, Vec3& v)
{
    in >> v.m_data[0] >> v.m_data[1] >> v.m_data[2];
    return in;
}

inline std::ostream& operator<<(std::ostream& out, const Vec3& v)
{
    out << v.m_data[0] << " " << v.m_data[1] << " " << v.m_data[2];
    return out;
}

__host__ __device__ constexpr inline void Vec3::normalize()
{
    const float value{ 1.f / length() };

    m_data[0] *= value;
    m_data[1] *= value;
    m_data[2] *= value;
}

__host__ __device__ constexpr inline Vec3 operator+(const Vec3& lhs, const Vec3& rhs)
{
    return { lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z() };
}

__host__ __device__ constexpr inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs)
{
    return { lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z() };
}

__host__ __device__ constexpr inline Vec3 operator*(const Vec3& lhs, const Vec3& rhs)
{
    return { lhs.x() * rhs.x(), lhs.y() * rhs.y(), lhs.z() * rhs.z() };
}

__host__ __device__ constexpr inline Vec3 operator/(const Vec3& lhs, const Vec3& rhs)
{
    return { lhs.x() / rhs.x(), lhs.y() / rhs.y(), lhs.z() / rhs.z() };
}

__host__ __device__ constexpr inline Vec3 operator*(float lhs, const Vec3& rhs)
{
    return { lhs * rhs.x(), lhs * rhs.y(), lhs * rhs.z() };
}

__host__ __device__ constexpr inline Vec3 operator/(const Vec3& lhs, float rhs)
{
    return { lhs.x() / rhs, lhs.y() / rhs, lhs.z() / rhs };
}

__host__ __device__ constexpr inline Vec3 operator*(const Vec3& lhs, float rhs)
{
    return { lhs.x() * rhs, lhs.y() * rhs, lhs.z() * rhs };
}

__host__ __device__ constexpr inline float dot(const Vec3& lhs, const Vec3& rhs)
{
    return (lhs.x() * rhs.x()) + (lhs.y() * rhs.y()) + (lhs.z() * rhs.z());
}

__host__ __device__ constexpr inline Vec3 cross(const Vec3& lhs, const Vec3& rhs)
{
    return { (lhs.y() * rhs.z()) - (lhs.z() * rhs.y()),
             -(lhs.x() * rhs.z()) - (lhs.z() * rhs.x()),
             (lhs.x() * rhs.y()) - (lhs.y() * rhs.x()) };
}

__host__ __device__ constexpr inline Vec3& Vec3::operator+=(const Vec3& other)
{
    m_data[0] += other.x();
    m_data[1] += other.y();
    m_data[2] += other.z();

    return *this;
}

__host__ __device__ constexpr inline Vec3& Vec3::operator*=(const Vec3& other)
{
    m_data[0] *= other.x();
    m_data[1] *= other.y();
    m_data[2] *= other.z();

    return *this;
}

__host__ __device__ constexpr inline Vec3& Vec3::operator/=(const Vec3& other)
{
    m_data[0] /= other.x();
    m_data[1] /= other.y();
    m_data[2] /= other.z();

    return *this;
}

__host__ __device__ constexpr inline Vec3& Vec3::operator-=(const Vec3& other)
{
    m_data[0] -= other.x();
    m_data[1] -= other.y();
    m_data[2] -= other.z();

    return *this;
}

__host__ __device__ constexpr inline Vec3& Vec3::operator*=(float scalar)
{
    m_data[0] *= scalar;
    m_data[1] *= scalar;
    m_data[2] *= scalar;

    return *this;
}

__host__ __device__ constexpr inline Vec3& Vec3::operator/=(float scalar)
{
    const auto value{ 1.f / scalar };

    m_data[0] *= value;
    m_data[1] *= value;
    m_data[2] *= value;

    return *this;
}

__host__ __device__ constexpr inline Vec3 unitVector(const Vec3& v)
{
    return v / v.length();
}

#endif // !CRT_SRC_VEC_3_HPP
