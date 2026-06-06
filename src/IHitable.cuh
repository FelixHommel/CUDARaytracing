#ifndef CRT_SRC_I_HITABLE_CUH
#define CRT_SRC_I_HITABLE_CUH

#include "src/Ray.cuh"
#include "src/Vec3.cuh"

/// \brief Document a hit between a \ref Ray and a \ref IHitable.
///
/// \author Felix Hommel
/// \date 6/6/2026
struct HitRecord
{
    float t{ 0.f };
    Vec3 p;
    Vec3 normal;
};

/// \brief Represent any object that can intersect with a \ref Ray.
///
/// \author Felix Hommel
/// \date 6/6/2026
class IHitable // NOLINT
{
public:
    IHitable() = default;
    __device__ virtual ~IHitable() = default;

    __device__ virtual bool hit(const Ray& r, float min, float max, HitRecord& rec) const = 0;
};

#endif // !CRT_SRC_I_HITABLE_CUH
