cmake_minimum_required(VERSION 3.31)
project(raytracing LANGUAGES CUDA CXX C)

find_package(fmt CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(${PROJECT_NAME} ./src/main.cu)
target_sources(
    ${PROJECT_NAME}
    PRIVATE
        FILE_SET HEADERS
            FILES
                ./src/Vec3.cuh
                ./src/Ray.cuh
                ./src/IHitable.cuh
                ./src/HitableList.cuh
                ./src/Sphere.cuh
                ./src/Camera.cuh
                ./src/Material.cuh
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20 cuda_std_20)

target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)
