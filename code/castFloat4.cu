#include "castFloat4.cuh"

#include "memcpy.cuh"

#include <cuda_runtime.h>


//--------------------------------------------------------------------------
// Declare the interleaved copu CUDA kernel
//--------------------------------------------------------------------------
template<typename T> __global__ void CopyCastInterleaved(unsigned char* destination, const T* source, unsigned int pitch, unsigned int width)
{
	uint2 index = make_uint2(
		__umul24(blockIdx.x, blockDim.x) + threadIdx.x,
		__umul24(blockIdx.y, blockDim.y) + threadIdx.y);
	unsigned int index3 = 3 * (index.y * width + index.x);
	
	float4* dest = (float4*)(destination + index.y * pitch) + index.x;
	float mult = 1.0f / Multiplier<T>();
	*dest = make_float4(
		mult * (float)source[index3],
		mult * (float)source[index3+1],
		mult * (float)source[index3+2], 1.0f);
}

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
template<typename T> cudaPitchedPtr CastVolumeHost3ToDevice4(const T* host, unsigned int width, unsigned int height, unsigned int depth)
{
	cudaPitchedPtr device = {0};
	const cudaExtent extent = make_cudaExtent(width * sizeof(float4), height, depth);
	CUDA_SAFE_CALL(cudaMalloc3D(&device, extent));
	const size_t pitchedBytesPerSlice = device.pitch * device.ysize;
	
	T* temp = 0;
	const unsigned int voxelsPerSlice = width * height;
	const size_t nrOfBytesTemp = voxelsPerSlice * 3 * sizeof(T);
	CUDA_SAFE_CALL(cudaMalloc((void**)&temp, nrOfBytesTemp));

	unsigned int dimX = min(PowTwoDivider(width), 64);
	dim3 dimBlock(dimX, min(PowTwoDivider(height), 512 / dimX));
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
	size_t offsetHost = 0;
	size_t offsetDevice = 0;
	
	for (unsigned int slice = 0; slice < depth; slice++)
	{
		CUDA_SAFE_CALL(cudaMemcpy(temp, host + offsetHost, nrOfBytesTemp, cudaMemcpyHostToDevice));
		CopyCastInterleaved<T><<<dimGrid, dimBlock>>>((unsigned char*)device.ptr + offsetDevice, temp, (unsigned int)device.pitch, width);
		CUT_CHECK_ERROR("Cast kernel failed");
		offsetHost += voxelsPerSlice;
		offsetDevice += pitchedBytesPerSlice;
	}

	CUDA_SAFE_CALL(cudaFree(temp));  //free the temp GPU volume
	return device;
}