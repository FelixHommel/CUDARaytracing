#ifndef CRT_SRC_HITABLE_LIST_CUH
#define CRT_SRC_HITABLE_LIST_CUH

#include "IHitable.cuh"
#include "Ray.cuh"

/// \brief Container that manages multiple \ref IHitable.
///
/// \author Felix Hommel
/// \date 6/6/2026
struct HitableList : public IHitable // NOLINT
{
public:
    __device__ HitableList() {} // NOLINT
    __device__ HitableList(IHitable** l, int n) : list(l), listSize(n) {}
    __device__ ~HitableList() override {} // NOLINT

    /// \brief Check with which object in the List a \ref Ray intersects first (if any).
    ///
    /// \param r The \ref Ray that is being cast
    /// \param min Lower bounds of the hit interval
    /// \param max Upper bound sof the hit interval
    /// \param[out] rec Provide the caller with infos about the intersection that occured
    ///
    /// \returns \p true if \p r hits any object in this list, \p false otherwise
    __device__ bool hit(const Ray& r, float min, float max, HitRecord& rec) const override;

    IHitable** list;
    int listSize;
};

__device__ bool HitableList::hit(const Ray& r, float min, float max, HitRecord& rec) const
{
    HitRecord temp;
    bool hitAnything{ false };
    float closestSoFar{ max };

    for(int i{ 0 }; i < listSize; ++i)
    {
        if(list[i]->hit(r, min, closestSoFar, temp)) // NOLINT
        {
            hitAnything = true;
            closestSoFar = temp.t;
            rec = temp;
        }
    }

    return hitAnything;
}

#endif // !CRT_SRC_HITABLE_LIST_CUH
