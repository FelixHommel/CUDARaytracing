#ifndef CRT_SRC_MATERIAL_CUH
#define CRT_SRC_MATERIAL_CUH

#include "src/IHitable.cuh"
#include "src/Ray.cuh"
#include "src/Vec3.cuh"

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <cmath>

/// \brief Generate a \ref Vec3 with random values
///
/// \param randState The \ref curandState to access the thread local random state
///
/// \returns \ref Vec3 with random x, y, and z components
__device__ inline Vec3 randVec3(curandState* randState)
{
    return { curand_uniform(randState), curand_uniform(randState), curand_uniform(randState) };
}

__device__ Vec3 randomInUnitSphere(curandState* localRandState)
__device__ Vec3 randomInUnitSphere(curandState* randState)
{
    Vec3 p{};

    do // NOLINT
    {
        p = (2.f * randVec3(randState)) - Vec3{ 1.f };
    }
    while(p.lengthSquared() >= 1.f);

    return p;
}

__device__ Vec3 reflect(const Vec3& v, const Vec3& n)
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

class IMaterial // NOLINT
{
public:
    __device__ IMaterial() {}          // NOLINT
    __device__ virtual ~IMaterial() {} // NOLINT

    __device__ virtual bool scatter(
        const Ray& in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState
    ) const = 0;
};

struct Lambertian : IMaterial // NOLINT
{
    __device__ Lambertian(const Vec3& a) : albedo(a) {}
    __device__ ~Lambertian() override {} // NOLINT

    __device__ bool scatter(
        const Ray& /*in*/, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState
    ) const override
    {
        const Vec3 target{ rec.p + rec.normal + randomInUnitSphere(localRandState) };
        scattered = { rec.p, target - rec.p };
        attenuation = albedo;

        return true;
    }

    Vec3 albedo;
};

struct Metal : IMaterial // NOLINT
{
    __device__ Metal(const Vec3& a, float f) : albedo(a), fuzz(f) {}
    __device__ ~Metal() override {} // NOLINT

    __device__ bool scatter(
        const Ray& in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState
    ) const override
    {
        const Vec3 reflected{ reflect(unitVector(in.direction), rec.normal) };
        scattered = { rec.p, reflected + (fuzz * randomInUnitSphere(localRandState)) };
        attenuation = albedo;

        return (dot(scattered.direction, rec.normal) > 0.f);
    }

    Vec3 albedo;
    float fuzz;
};

struct Dielectric : IMaterial // NOLINT
{
    __device__ Dielectric(float ri) : refractIndex(ri) {}
    __device__ ~Dielectric() override {} // NOLINT

    __device__ bool scatter(
        const Ray& in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState
    ) const override
    {
        const Vec3 reflected{ reflect(in.direction, rec.normal) };

        Vec3 outwardNormal;
        float niOverNt{ 0.f };
        attenuation = Vec3{ 1.f };
        Vec3 refracted;
        float reflectProbability{ 0.f };
        float cosine{ 0.f };

        if(dot(in.direction, rec.normal) > 0.f)
        {
            outwardNormal = -rec.normal;
            niOverNt = refractIndex;
            cosine = dot(in.direction, rec.normal) / in.direction.length();
            cosine = std::sqrt(1.f - ((refractIndex * refractIndex) * (1.f - (cosine * cosine))));
        }
        else
        {
            outwardNormal = rec.normal;
            niOverNt = 1.f / refractIndex;
            cosine = -dot(in.direction, rec.normal) / in.direction.length();
        }

        if(refract(in.direction, outwardNormal, niOverNt, refracted))
            reflectProbability = schlick(cosine, refractIndex);
        else
            reflectProbability = 1.f;

        if(curand_uniform(localRandState) < reflectProbability)
            scattered = { rec.p, reflected };
        else
            scattered = { rec.p, refracted };

        return true;
    }

    float refractIndex;
};

#endif // !CRT_SRC_MATERIAL_CUH
