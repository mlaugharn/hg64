cmake_minimum_required(VERSION 3.14...3.22)
project(hg64 LANGUAGES CXX C)
# set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH}
# "${PROJECT_SOURCE_DIR}/scikit-build/skbuild/resources/cmake")
find_package(PythonExtensions REQUIRED)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)
include(FetchContent)

FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.tar.gz
  URL_HASH
    SHA256=6bd528c4dbe2276635dc787b6b1f2e5316cf6b49ee3e150264e455a0d68d19c1)
FetchContent_MakeAvailable(pybind11)
set(python_module_name _hg64)

set(HG64_SOURCES src/hg64/hg64.cpp src/hg64/hg64.h src/hg64/random.cpp
                 src/hg64/random.h)

pybind11_add_module(${python_module_name} MODULE ${HG64_SOURCES})
install(TARGETS ${python_module_name} DESTINATION .)

target_link_libraries(${python_module_name} PRIVATE)
if(MSVC)
  add_compile_options("/W4" "/O2" "/LTCG" "/MD")
else()
  add_compile_options("-Wall" "-Wextra" "-Werror" "$<$<CONFIG:RELEASE>:-O3>")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options("-stdlib=libc++")
  else()
    # nothing special for gcc at the moment
  endif()
endif()
set(ignoreMe "${SKBUILD}")
