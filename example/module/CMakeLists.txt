cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(modules LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Debug mode
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(run PRIVATE DEBUG_MODE)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_definitions(run PRIVATE RELEASE_MODE)
endif()

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set warning flags
add_compile_options(
  -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare
)

# use -std=c++20 for clang
# if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
add_compile_options(-std=c++20 -stdlib=libc++ -fmodules -fbuiltin-module-map)

add_executable(main helloworld.cpp main.cpp)

# target_sources(main PRIVATE
#     FILE_SET cxx_modules TYPE CXX_MODULES FILES
#     ${CMAKE_CURRENT_SOURCE_DIR}/helloworld.cpp
# )

# target_link_libraries(main PRIVATE helloworld)


