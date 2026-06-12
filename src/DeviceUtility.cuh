#ifndef CRT_SRC_DEVICE_UTILITY_CUH
#define CRT_SRC_DEVICE_UTILITY_CUH

#include "src/Vec3.cuh"

#include <curand_kernel.h>

namespace device
{

/// \brief Generate a random number.
///
/// \param randState The \ref curandState to access the thread local random state
///
/// \returns the generated random number
__device__ inline float randNum(curandStatePhilox4_32_10_t* randState)
{
    return curand_uniform(randState);
}

/// \brief Generate a \ref Vec3 with random values
///
/// \param randState The \ref curandState to access the thread local random state
///
/// \returns \ref Vec3 with random x, y, and z components
__device__ inline Vec3 randVec3(curandStatePhilox4_32_10_t* randState)
{
    return { randNum(randState), randNum(randState), randNum(randState) };
}

/// \brief Generate a random point within a unit sphere.
///
/// \param randState The \ref curandState to access the thread local random state
///
/// \returns \ref Vec3 the point that lies somewhere within the unit sphere
__device__ Vec3 randomInUnitSphere(curandStatePhilox4_32_10_t* randState)
{
    Vec3 p{};

    do // NOLINT
    {
        p = (2.f * randVec3(randState)) - Vec3{ 1.f };
    }
    while(p.lengthSquared() >= 1.f);

    return p;
}

/// \brief
///
/// \param randState The \ref curandState to access the thread local random state
///
/// \returns \ref Vec3
__device__ Vec3 randomInUnitDisk(curandStatePhilox4_32_10_t* randState)
{
    Vec3 p{};
    do // NOLINT
    {
        p = 2.f * Vec3{ device::randNum(randState), device::randNum(randState), 0.f } - Vec3{ 1.f, 1.f, 0.f };
    }
    while(dot(p, p) >= 1.f);

    return p;
}

__device__ inline Vec3 reflect(const Vec3& v, const Vec3& n)
{
    return v - (2.f * dot(v, n) * n);
}

__device__ bool refract(const Vec3& v, const Vec3& n, float niOverNt, Vec3& refracted)
{
    const auto uv{ unitVector(v) };
    const auto dt{ dot(uv, n) };
    const float discriminant{ 1.f - ((niOverNt * niOverNt) * (1.f - (dt * dt))) };

    if(discriminant > 0.f)
    {
        refracted = (niOverNt * (uv - (n * dt)) - (n * std::sqrt(discriminant)));
        return true;
    }

    return false;
}

__device__ float schlick(float cosine, float refractIndex)
{
    float r0{ (1.f - refractIndex) / (1.f + refractIndex) };
    r0 = r0 * r0;

    return r0 + ((1.f - r0) * std::pow(1.f - cosine, 5.f)); // NOLINT
}


} // namespace device

#endif // !CRT_SRC_DEVICE_UTILITY_CUH
