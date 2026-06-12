#ifndef CRT_SRC_RAY_TRACER_CUH
#define CRT_SRC_RAY_TRACER_CUH

#include "src/Camera.cuh"
#include "src/HitableList.cuh"
#include "src/IHitable.cuh"
#include "src/Material.cuh"
#include "src/Ray.cuh"
#include "src/Sphere.cuh"
#include "src/Utility.cuh"
#include "src/Vec3.cuh"

#include <curand_kernel.h>

#include <cfloat>

static constexpr auto OBJECTS_IN_SCENE{ (22u * 22u) + 1u + 3u }; ///< Determine How many objects are in the scene

namespace
{

__device__ constexpr auto DEV_RANDOM_SEED{ 0xC0FFEE };

__device__ constexpr auto DEV_MAX_COLOR_ITERATIONS{ 50u };
__device__ constexpr auto DEV_HIT_MIN{ 0.001f };
__device__ constexpr auto DEV_COLOR_FACTOR{ Vec3(0.5f, 0.7f, 1.f) };

} // namespace

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic):
namespace tracer
{

/// \brief Initialize the random number generator that is used to generate the World.
///
/// \param randState The \ref curandState to access the thread local random state
__global__ void worldRandInit(curandState* randState)
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
        curand_init(::DEV_RANDOM_SEED, 0, 0, randState);
}

/// \brief Kernel to prepare the rendering state on the GPU.
///
/// \param width The width of the framebuffer
/// \param height The height of the framebuffer
/// \param randState The \ref curandState to access the thread local random state
///
/// \note CUDA Kernel
__global__ void renderInit(int width, int height, curandStatePhilox4_32_10_t* randState)
{
    const auto i{ threadIdx.x + (blockIdx.x * blockDim.x) };
    const auto j{ threadIdx.y + (blockIdx.y * blockDim.y) };

    if(i >= width || j >= height)
        return;

    const auto pixelIndex{ shared::calculatePixelIndex(i, j, width) };

    curand_init(::DEV_RANDOM_SEED, pixelIndex, 0, &randState[pixelIndex]);
}

// NOLINTBEGIN(readability-magic-numbers): It's fine for the init of the world

/// \brief Kernel to create the objects that are in the world on the GPU.
///
/// \param list The objects that are in the world
/// \param world The container that contains the objects in \p list
/// \param camera The \ref Camera that observes the scene
/// \param width The width of the framebuffer
/// \param height The height of the framebuffer
/// \param randState The \ref curandState to access the thread local random state
///
/// \note CUDA Kernel
__global__ void createWorld(
    IHitable** list, IHitable** world, Camera** camera, int width, int height, curandState* randState
)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    auto localRandState{ *randState };

    list[0] = new Sphere{
        Vec3{ 0.f, -1000.f, -1.f },
        1000.f, new Lambertian{ Vec3{ 0.5f } }
    };

    int i{ 1 };
    for(int a{ -11 }; a < 11; ++a)
    {
        for(int b{ -11 }; b < 11; ++b)
        {
            const float chooseMat{ device::randNum(&localRandState) };
            const Vec3 center{
                static_cast<float>(a) + device::randNum(&localRandState),
                0.2f,
                static_cast<float>(b) + device::randNum(&localRandState),
            };

            if(chooseMat < 0.8f)
            {
                list[i++] = new Sphere{
                    center,
                    0.2f,
                    new Lambertian{ Vec3{
                        device::randNumProduct(&localRandState),
                        device::randNumProduct(&localRandState),
                        device::randNumProduct(&localRandState),
                    } },
                };
            }
            else if(chooseMat < 0.95f)
            {
                const auto metalColorFn{ [&localRandState] {
                    return 0.5f * (1.f + device::randNum(&localRandState));
                } };

                list[i++] = new Sphere{
                    center,
                    0.2f,
                    new Metal{ Vec3{ metalColorFn(), metalColorFn(), metalColorFn() },
                              0.5f * device::randNum(&localRandState) }
                };
            }
            else
                list[i++] = new Sphere{ center, 0.2f, new Dielectric{ 1.5f } };
        }
    }

    list[i++] = new Sphere{
        Vec3{ 0.f, 1.f, 0.f },
        1.f, new Dielectric{ 1.5f }
    };
    list[i++] = new Sphere{
        Vec3{ -4.f, 1.f, 0.f },
        1.f, new Lambertian{ Vec3{ 0.4f, 0.2f, 0.1f } }
    };
    list[i++] = new Sphere{
        Vec3{ 4.f, 1.f, 0.f },
        1.f, new Metal{ Vec3{ 0.7f, 0.6f, 0.5f }, 0.f }
    };

    *randState = localRandState;

    *world = new HitableList(list, ::OBJECTS_IN_SCENE);

    Vec3 lookFrom{ 13.f, 2.f, 3.f };
    Vec3 lookAt{ 0.f, 0.f, 0.f };
    float distanceToFocus{ 10.f };
    float aperture{ 0.1f };
    *camera = new Camera(
        lookFrom,
        lookAt,
        Vec3(0.f, 1.f, 0.f),
        30.f,
        (static_cast<float>(width) / static_cast<float>(height)),
        aperture,
        distanceToFocus
    );
}

// NOLINTEND(readability-magic-numbers)

/// \brief Kernel to destroy the objects that are in the world on the GPU.
///
/// \param list The objects that are in the world
/// \param world The container that contains the objects in \p list
/// \param camera The \ref Camera that observes the scene
///
/// \note CUDA Kernel
__global__ void freeWorld(IHitable** list, IHitable** world, Camera** camera)
{
    for(int i{ 0 }; i < ::OBJECTS_IN_SCENE; ++i)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast): dynamic_cast is not allowed in device code, therefore static_cast is the only option
        delete static_cast<Sphere*>(list[i])->pMaterial;
        delete list[i];
    }
    delete *world;
    delete *camera;
}

/// \brief Calculate the color at a specific \ref Ray position.
///
/// \param r The ray that points to a position
/// \param world List of all objects in the world
/// \param localRandState The thread local \ref curandState
///
/// \returns Vec3 Color at that position
///
/// \note Only callable from a CUDA Kernel or other device functions.
__device__ Vec3 color(const Ray& r, IHitable** world, curandStatePhilox4_32_10_t* localRandState)
{
    Ray curRay{ r };
    Vec3 curAttenuation{ 1.f };

    for(int i{ 0 }; i < ::DEV_MAX_COLOR_ITERATIONS; ++i)
    {
        HitRecord rec;
        if((*world)->hit(curRay, ::DEV_HIT_MIN, FLT_MAX, rec))
        {
            Ray scattered;
            Vec3 attenuation;

            if(rec.pMaterial->scatter(curRay, rec, attenuation, scattered, localRandState))
            {
                curAttenuation *= attenuation;
                curRay = scattered;
            }
            else
                return Vec3{ 0.f };
        }
        else
        {
            const Vec3 dir{ unitVector(r.direction) };
            const float t{ 0.5f * (dir.y() + 1.f) };
            const Vec3 c{ (1.f - t) * Vec3(1.f) + t * ::DEV_COLOR_FACTOR };

            return curAttenuation * c;
        }
    }

    return Vec3{ 0.f };
}

/// \brief Kernel that produces the framebuffer.
///
/// \param pFramebuffer The framebuffer array
/// \param width The width of the framebuffer
/// \param height The height of the framebuffer
/// \param sampleCount The amount of samples taken per pixel
/// \param camera The \ref Camera that observes the scene
/// \param world List of all objects in the world
/// \param randState The \ref curandState to access the thread local random state
///
/// \note CUDA Kernel
__global__ void render(
    Vec3* pFramebuffer,
    int width,
    int height,
    unsigned int sampleCount,
    Camera** camera,
    IHitable** world,
    curandStatePhilox4_32_10_t* randState
)
{
    const auto i{ threadIdx.x + (blockIdx.x * blockDim.x) };
    const auto j{ threadIdx.y + (blockIdx.y * blockDim.y) };

    if(i >= width || j >= height)
        return;

    const auto pixelIndex{ shared::calculatePixelIndex(i, j, width) };
    auto localRandState{ randState[pixelIndex] };

    auto c{ Vec3{ 0.f } };
    for(int s{ 0 }; s < sampleCount; ++s)
    {
        const auto u{ (static_cast<float>(i) + device::randNum(&localRandState)) / static_cast<float>(width) };
        const auto v{ (static_cast<float>(j) + device::randNum(&localRandState)) / static_cast<float>(height) };

        const auto r{ (*camera)->getRay(u, v, &localRandState) };
        c += tracer::color(r, world, &localRandState);
    }

    randState[pixelIndex] = localRandState;

    // NOTE: Simplified color correction
    c /= static_cast<float>(sampleCount);
    c[0] = std::sqrt(c[0]);
    c[1] = std::sqrt(c[1]);
    c[2] = std::sqrt(c[2]);

    pFramebuffer[pixelIndex] = c;
}

} // namespace tracer
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

#endif // !CRT_SRC_RAY_TRACER_CUH
