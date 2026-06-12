#ifndef CRT_SRC_I_HITABLE_CUH
#define CRT_SRC_I_HITABLE_CUH

class IMaterial;

#include "Ray.cuh"
#include "Vec3.cuh"

/// \brief Document a hit between a \ref Ray and a \ref IHitable.
///
/// \author Felix Hommel
/// \date 6/6/2026
struct HitRecord
{
    float t{ 0.f };
    Vec3 p;
    Vec3 normal;
    IMaterial* pMaterial{ nullptr };
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
