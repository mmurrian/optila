cmake_minimum_required(VERSION 3.10)

project(OPTiLA LANGUAGES CXX)

# Header-only library target
add_library(OPTiLA INTERFACE)
target_include_directories(
  OPTiLA INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                   $<INSTALL_INTERFACE:include>)

option(OPTILA_ENABLE_IMPLICIT_CONVERSIONS
       "Enable implicit conversions when evaluating OPTiLA expressions" OFF)
if(OPTILA_ENABLE_IMPLICIT_CONVERSIONS)
  message(STATUS "OPTiLA: Implicit conversions enabled")
  target_compile_definitions(OPTiLA
                             INTERFACE OPTILA_ENABLE_IMPLICIT_CONVERSIONS)
else()
  message(STATUS "OPTiLA: Implicit conversions disabled")
endif()

# Set C++17 standard for the OPTiLA target
target_compile_features(OPTiLA INTERFACE cxx_std_17)

# Add header files
target_sources(
  OPTiLA
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/optila.h>
            # Add more header files here
)

# Executable for development, testing, and compiling against OPTiLA
add_executable(OPTiLATest main.cpp)

# Set C++17 standard for the OPTiLATest target
target_compile_features(OPTiLATest PRIVATE cxx_std_17)

target_link_libraries(OPTiLATest PRIVATE OPTiLA)

# Install the OPTiLA library and headers
install(
  TARGETS OPTiLA
  EXPORT OPTiLATargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

install(
  EXPORT OPTiLATargets
  FILE OPTiLATargets.cmake
  NAMESPACE OPTiLA::
  DESTINATION lib/cmake/OPTiLA)

install(
  FILES include/optila.h
        include/details/optila_matrix.h
        include/details/optila_type_traits.h
        include/details/optila_scalar.h
        include/details/optila_expression.h
        include/details/optila_operation.h
        include/optila_matrix_impl.h
        include/optila_scalar_impl.h
        include/optila_expression_impl.h
        include/optila_expression_validator_impl.h
        include/optila_operation_impl.h
        include/optila_wrapper_free_functions.h
        include/optila_wrapper_operator_overloads.h
        include/optila_evaluate_impl.h
        include/optila_wrapper_aliases.h
  DESTINATION include)
