#ifndef CRT_SRC_MATERIAL_CUH
#define CRT_SRC_MATERIAL_CUH

#include "src/IHitable.cuh"
#include "src/Ray.cuh"
#include "src/Utility.cuh"
#include "src/Vec3.cuh"

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <cmath>

/// \brief Interface description that any Material type should adhere to.
///
/// \author Felix Hommel
/// \date 6/6/2026
class IMaterial // NOLINT
{
public:
    __device__ IMaterial() {} // NOLINT
    __device__ virtual ~IMaterial() = default;

    __device__ virtual bool scatter(
        const Ray& in,
        const HitRecord& rec,
        Vec3& attenuation,
        Ray& scattered,
        curandStatePhilox4_32_10_t* localRandState
    ) const = 0;
};

/// \brief Material that has a surface that implements the Lambertian reflectance model.
///
/// \author Felix Hommel
/// \date 6/6/2026
class Lambertian : public IMaterial // NOLINT
{
public:
    __device__ Lambertian(const Vec3& a) : m_albedo(a) {}
    __device__ ~Lambertian() override = default;

    __device__ bool scatter(
        const Ray& /*in*/,
        const HitRecord& rec,
        Vec3& attenuation,
        Ray& scattered,
        curandStatePhilox4_32_10_t* localRandState
    ) const override
    {
        const Vec3 target{ rec.p + rec.normal + device::randomInUnitSphere(localRandState) };
        scattered = { rec.p, target - rec.p };
        attenuation = m_albedo;

        return true;
    }

private:
    Vec3 m_albedo;
};

/// \brief Material that has a surface that implements Metal reflective properties.
///
/// \author Felix Hommel
/// \date 6/7/2026
class Metal : public IMaterial // NOLINT
{
public:
    __device__ Metal(const Vec3& a, float f) : m_albedo(a), m_fuzz(f) {}
    __device__ ~Metal() override = default;

    __device__ bool scatter(
        const Ray& in,
        const HitRecord& rec,
        Vec3& attenuation,
        Ray& scattered,
        curandStatePhilox4_32_10_t* localRandState
    ) const override
    {
        const Vec3 reflected{ device::reflect(unitVector(in.direction), rec.normal) };
        scattered = { rec.p, reflected + (m_fuzz * device::randomInUnitSphere(localRandState)) };
        attenuation = m_albedo;

        return (dot(scattered.direction, rec.normal) > 0.f);
    }

private:
    Vec3 m_albedo;
    float m_fuzz;
};

/// \brief Material that has a surface that implements Dielectric reflective properties.
///
/// \author Felix Hommel
/// \date 6/8/2026
class Dielectric : public IMaterial // NOLINT
{
public:
    __device__ Dielectric(float ri) : m_refractIndex(ri) {}
    __device__ ~Dielectric() override = default;

    __device__ bool scatter(
        const Ray& in,
        const HitRecord& rec,
        Vec3& attenuation,
        Ray& scattered,
        curandStatePhilox4_32_10_t* localRandState
    ) const override
    {
        const Vec3 reflected{ device::reflect(in.direction, rec.normal) };

        Vec3 outwardNormal;
        float niOverNt{ 0.f };
        attenuation = Vec3{ 1.f };
        Vec3 refracted;
        float reflectProbability{ 0.f };
        float cosine{ 0.f };

        if(dot(in.direction, rec.normal) > 0.f)
        {
            outwardNormal = -rec.normal;
            niOverNt = m_refractIndex;
            cosine = dot(in.direction, rec.normal) / in.direction.length();
            cosine = std::sqrt(1.f - ((m_refractIndex * m_refractIndex) * (1.f - (cosine * cosine))));
        }
        else
        {
            outwardNormal = rec.normal;
            niOverNt = 1.f / m_refractIndex;
            cosine = -dot(in.direction, rec.normal) / in.direction.length();
        }

        if(device::refract(in.direction, outwardNormal, niOverNt, refracted))
            reflectProbability = device::schlick(cosine, m_refractIndex);
        else
            reflectProbability = 1.f;

        if(device::randNum(localRandState) < reflectProbability)
            scattered = { rec.p, reflected };
        else
            scattered = { rec.p, refracted };

        return true;
    }

private:
    float m_refractIndex;
};

#endif // !CRT_SRC_MATERIAL_CUH
