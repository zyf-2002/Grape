cmake_minimum_required(VERSION 3.18)
set(CMAKE_BUILD_TYPE Debug)

project(lightZK LANGUAGES CXX CUDA)
# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Set CUDA Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --extended-lambda -lineinfo")

# Set Searching Path for Header Files
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE CU_FILES "${PROJECT_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE CPP_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(zkLLM STATIC ${CPP_FILES} ${CU_FILES})


set_target_properties(zkLLM PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(zkLLM PUBLIC snark)

target_include_directories(zkLLM PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/depends/libsnark
)

set(CURVE "ALT_BN128" CACHE STRING "Default curve: one of ALT_BN128, BN128, EDWARDS, MNT4, MNT6")
add_definitions(-DCURVE_${CURVE})
add_subdirectory(depends/libsnark)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fopenmp")

# 编译测试程序                                                                                                                                                                                  
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/test/*.cu")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE 
		zkLLM
		snark
	)
    target_include_directories(${test_name} PRIVATE 
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_SOURCE_DIR}/depends/libsnark
	)
endforeach()
