#pragma once

#include <host_defines.h>
#include "Prefix.h"

//--------------------------------------------------------------------------
// Declare the typecast CUDA kernels
//--------------------------------------------------------------------------
template<typename T> __device__ float Multiplier()	{ return 1.0f; }

//--------------------------------------------------------------------------
// Declare the typecast templated function
// This function can be called directly in C++ programs
//--------------------------------------------------------------------------

//! Allocate GPU memory and copy a voxel volume from CPU to GPU memory
//! and cast it to the normalized floating point format
//! @return the pointer to the GPU copy of the voxel volume
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
template<typename T> 
cudaPitchedPtr CastVolumeHostToDevice(const T* host, unsigned int width, unsigned int height, unsigned int depth);