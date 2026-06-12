#ifndef CRT_SRC_RAY_CUH
#define CRT_SRC_RAY_CUH

#include "Vec3.cuh"

#include <cuda_runtime_api.h>

/// \brief CUDA device class that can represent a euclidean vector.
///
/// \author Felix Hommel
/// \date 6/5/2026
struct Ray
{
public:
    __device__ Ray() {} // NOLINT(modernize-use-equals-default): nvcc warning: non-virtual __device__ = default methods
    __device__ Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}

    __device__ Vec3 pointAtParameter(float t) const { return origin + (t * direction); }

    Vec3 origin;
    Vec3 direction;
};

#endif // !CRT_SRC_RAY_CUH
