set(NAME "CRTBenchmark")

add_executable(${NAME} ./main.cu)
target_sources(
    ${NAME}
    PRIVATE
        FILE_SET HEADERS
            FILES
                ../src/Vec3.cuh
                ../src/Ray.cuh
                ../src/IHitable.cuh
                ../src/HitableList.cuh
                ../src/Sphere.cuh
                ../src/Camera.cuh
                ../src/Material.cuh
                ../src/Utility.cuh
                ../src/RayTracer.cuh
                ../src/Error.cuh
)

target_compile_features(${NAME} PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(${NAME} PRIVATE fmt::fmt)
