cmake_minimum_required(VERSION 3.20)

# Prebuilt mode only installs files, so do not require a compiler at the top
# level. The source subproject enables C++ and CUDA when it is selected.
project(PySynrealSolver LANGUAGES NONE)

# Auto set default value for SYNREAL_SOLVER_USE_PREBUILT
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/synreal-solver/CMakeLists.txt")
    set(_SYNREAL_SOLVER_PREBUILT_DEFAULT OFF)
else()
    set(_SYNREAL_SOLVER_PREBUILT_DEFAULT ON)
endif()

# CMake build option
option(
    SYNREAL_SOLVER_USE_PREBUILT
    "Install the extension from style3d-sim-bin instead of building it from source"
    ${_SYNREAL_SOLVER_PREBUILT_DEFAULT}
)

# Add subdirectory
if(SYNREAL_SOLVER_USE_PREBUILT)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/style3d-sim-bin/CMakeLists.txt")
        message(FATAL_ERROR "SYNREAL_SOLVER_USE_PREBUILT=ON, but style3d-sim-bin is not initialized.")
    endif()
    add_subdirectory(style3d-sim-bin)
else()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/synreal-solver/CMakeLists.txt")
        message(FATAL_ERROR "SYNREAL_SOLVER_USE_PREBUILT=OFF, but synreal-solver is not initialized.")
    endif()
    # Select the Python package layout without excluding the native targets from the default build.
    set(SYNREAL_SOLVER_PYTHON_INSTALL_DIR style3d_sim)
    add_subdirectory(synreal-solver)
endif()
