#ifndef CRT_SRC_SPHERE_CUH
#define CRT_SRC_SPHERE_CUH

#include "src/IHitable.cuh"
#include "src/Ray.cuh"
#include "src/Vec3.cuh"

#include <cmath>

/// \brief A Spherical object that can be hit by a \ref Ray.
///
/// \author Felix Hommel
/// \date 6/6/2026
struct Sphere : public IHitable // NOLINT
{
    __device__ Sphere() {} // NOLINT
    __device__ Sphere(Vec3 center, float radius) : center(center), radius(radius) {}
    __device__ ~Sphere() override {} // NOLINT

    /// \brief Check if a \ref Ray intersects with the \ref Sphere.
    ///
    /// \param r The \ref Ray that is being cast
    /// \param min Lower bounds of the hit interval
    /// \param max Upper bound sof the hit interval
    /// \param[out] rec Provide the caller with infos about the intersection that occured
    ///
    /// \returns \p true if \p r hits the \ref Sphere, \p false otherwise
    __device__ bool hit(const Ray& r, float min, float max, HitRecord& rec) const override;

    Vec3 center;
    float radius;
};

__device__ bool Sphere::hit(const Ray& r, float min, float max, HitRecord& rec) const
{
    const Vec3 oc{ r.origin - center };
    const float a{ dot(r.direction, r.direction) };
    const float b{ dot(oc, r.direction) };
    const float c{ dot(oc, oc) - (radius * radius) };
    const float discriminant{ (b * b) - (a * c) };

    if(discriminant > 0.f)
    {
        const float sqrtDiscriminant{ std::sqrt(discriminant) };

        float temp{ (-b - sqrtDiscriminant) / a };
        if(temp < max && temp > min)
        {
            rec.t = temp;
            rec.p = r.pointAtParameter(rec.t);
            rec.normal = (rec.p - center) / radius;

            return true;
        }

        temp = (-b + sqrtDiscriminant) / a;
        if(temp < max && temp > min)
        {
            rec.t = temp;
            rec.p = r.pointAtParameter(rec.t);
            rec.normal = (rec.p - center) / radius;

            return true;
        }
    }

    return false;
}

#endif // CRT_SRC_SPHERE_CUH
