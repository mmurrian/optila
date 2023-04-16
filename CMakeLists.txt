cmake_minimum_required(VERSION 3.10)

project(OPTiLA LANGUAGES CXX)

# Header-only library target
add_library(OPTiLA INTERFACE)
target_include_directories(OPTiLA INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Set C++17 standard for the OPTiLA target
target_compile_features(OPTiLA INTERFACE cxx_std_17)
set_target_properties(
  OPTiLA
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED YES
             CXX_EXTENSIONS NO)

# Add header files
target_sources(OPTiLA INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/optila.h>
  # Add more header files here
)

# Executable for development, testing, and compiling against OPTiLA
add_executable(OPTiLATest main.cpp)

# Set C++17 standard for the OPTiLATest target
target_compile_features(OPTiLATest PRIVATE cxx_std_17)
set_target_properties(
  OPTiLATest
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED YES
             CXX_EXTENSIONS NO)


target_link_libraries(OPTiLATest PRIVATE OPTiLA)

# Install the OPTiLA library and headers
install(TARGETS OPTiLA EXPORT OPTiLATargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(EXPORT OPTiLATargets
  FILE OPTiLATargets.cmake
  NAMESPACE OPTiLA::
  DESTINATION lib/cmake/OPTiLA
)

install(FILES
  include/optila.h
  # Add more header files here
  DESTINATION include
)

