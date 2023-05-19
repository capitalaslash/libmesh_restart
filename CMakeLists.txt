cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(LibmeshRestart
  VERSION 0.1
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic")

include(FetchContent)
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master
)
FetchContent_MakeAvailable(fmt)

include(FeatureSummary)

find_package(MPI REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(libmesh REQUIRED IMPORTED_TARGET libmesh)

# workaround for bogus libmesh include path
get_target_property(
  LIBMESH_INCLUDE_DIRECTORIES PkgConfig::libmesh INTERFACE_INCLUDE_DIRECTORIES)
list(REMOVE_ITEM LIBMESH_INCLUDE_DIRECTORIES "/usr/include/tirpc")
set_property(TARGET PkgConfig::libmesh
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${LIBMESH_INCLUDE_DIRECTORIES})

add_executable(main)
target_sources(main
  # PUBLIC header.hpp
  PRIVATE main.cpp
)
target_link_libraries(main
  PUBLIC
    MPI::MPI_CXX
    PkgConfig::libmesh
    fmt::fmt
)

feature_summary(WHAT ALL)

