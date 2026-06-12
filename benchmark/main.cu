#include "Camera.cuh"
#include "Error.cuh"
#include "IHitable.cuh"
#include "RayTracer.cuh"
#include "Vec3.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fmt/base.h>

namespace
{

constexpr auto NX{ 1200u };                                   ///< Pixels on the X-Axis (width)
constexpr auto NY{ 600u };                                    ///< Pixels on the Y-Axis (height)
constexpr auto NUM_PIXELS{ NX * NY };                         ///< The amount of pixels in the framebuffer
constexpr auto FRAMEBUFFER_SIZE{ NUM_PIXELS * sizeof(Vec3) }; ///< Framebuffer size in bytes

constexpr auto TX{ 8 };                                        ///< Threads on the X-Axis
constexpr auto TY{ 8 };                                        ///< Threads on the Y-Axis
constexpr dim3 BLOCKS{ (::NX / ::TX) + 1, (::NY / ::TY) + 1 }; ///< The layout of the blocks for CUDA Kernels
constexpr dim3 THREADS{ ::TX, ::TY };                          ///< The layout of the threads for CUDA Kernels

constexpr auto SAMPLE_COUNT{ 100u };

constexpr auto COLOR_MAX{ 255.99 };

} // namespace

int main()
{
    Vec3* d_framebuffer{ nullptr };
    CHECK_CUDA_ERROR(cudaMalloc(&d_framebuffer, ::FRAMEBUFFER_SIZE));
    CHECK_CUDA_ERROR(cudaGetLastError());

    curandStatePhilox4_32_10_t* d_randState{ nullptr };
    CHECK_CUDA_ERROR(cudaMalloc(&d_randState, ::NUM_PIXELS * sizeof(curandStatePhilox4_32_10_t)));

    tracer::renderInit<<<::BLOCKS, ::THREADS>>>(::NX, ::NY, d_randState);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    curandState* d_randStateWorld{ nullptr };
    CHECK_CUDA_ERROR(cudaMalloc(&d_randStateWorld, sizeof(curandState)));

    tracer::worldRandInit<<<1, 1>>>(d_randStateWorld);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    IHitable** d_list{ nullptr };
    CHECK_CUDA_ERROR(cudaMalloc(&d_list, ::OBJECTS_IN_SCENE * sizeof(IHitable*)));
    IHitable** d_world{ nullptr };
    CHECK_CUDA_ERROR(cudaMalloc(&d_world, sizeof(IHitable*)));
    Camera** d_camera{ nullptr };
    CHECK_CUDA_ERROR(cudaMalloc(&d_camera, sizeof(Camera*)));
    tracer::createWorld<<<1, 1>>>(d_list, d_world, d_camera, ::NX, ::NY, d_randStateWorld);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    cudaEvent_t start{ nullptr };
    cudaEventCreate(&start);
    cudaEvent_t stop{ nullptr };
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tracer::render<<<::BLOCKS, ::THREADS>>>(d_framebuffer, ::NX, ::NY, ::SAMPLE_COUNT, d_camera, d_world, d_randState);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms{ 0.f };
    cudaEventElapsedTime(&ms, start, stop);

    fmt::println("Render time for a {}x{} image: {:.2f} ms", ::NX, ::NY, ms);

    tracer::freeWorld<<<1, 1>>>(d_list, d_world, d_camera);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaFree(static_cast<void*>(d_camera)));
    CHECK_CUDA_ERROR(cudaFree(static_cast<void*>(d_world)));
    CHECK_CUDA_ERROR(cudaFree(static_cast<void*>(d_list)));
    CHECK_CUDA_ERROR(cudaFree(d_randState));
    CHECK_CUDA_ERROR(cudaFree(d_randStateWorld));
    CHECK_CUDA_ERROR(cudaFree(d_framebuffer));

    cudaDeviceReset();

    return 0;
}
