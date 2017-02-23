#include "cubicPrefilter.cuh"

#include <stdio.h>
#include "cutil.h"
#include "internal/cubicPrefilter_kernel.cuh"

//--------------------------------------------------------------------------
// Global CUDA procedures
//--------------------------------------------------------------------------
template<typename floatN>
__global__ void SamplesToCoefficients3DX(
	floatN* volume,		// in-place processing
	unsigned int pitch,			// width in bytes
	unsigned int width,			// width of the volume
	unsigned int height,		// height of the volume
	unsigned int depth)			// depth of the volume
{
	// process lines in x-direction
	const unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int z = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int startIdx = (z * height + y) * pitch;

	floatN* ptr = (floatN*)((unsigned char*)volume + startIdx);
	ConvertToInterpolationCoefficients(ptr, width, sizeof(floatN));
}

template<typename floatN>
__global__ void SamplesToCoefficients3DY(
	floatN* volume,		// in-place processing
	unsigned int pitch,			// width in bytes
	unsigned int width,			// width of the volume
	unsigned int height,		// height of the volume
	unsigned int depth)			// depth of the volume
{
	// process lines in y-direction
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int z = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int startIdx = z * height * pitch;

	floatN* ptr = (floatN*)((unsigned char*)volume + startIdx);
	ConvertToInterpolationCoefficients(ptr + x, height, pitch);
}

template<typename floatN>
__global__ void SamplesToCoefficients3DZ(
	floatN* volume,		// in-place processing
	unsigned int pitch,			// width in bytes
	unsigned int width,			// width of the volume
	unsigned int height,		// height of the volume
	unsigned int depth)			// depth of the volume
{
	// process lines in z-direction
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int startIdx = y * pitch;
	const unsigned int slice = height * pitch;

	floatN* ptr = (floatN*)((unsigned char*)volume + startIdx);
	ConvertToInterpolationCoefficients(ptr + x, depth, slice);
}

//--------------------------------------------------------------------------
// Exported functions
//--------------------------------------------------------------------------

//! Convert the voxel values into cubic b-spline coefficients
//! @param volume  pointer to the voxel volume in GPU (device) memory
//! @param pitch   width in bytes (including padding bytes)
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
template<typename floatN>
void CubicBSplinePrefilter3D(floatN* volume, unsigned int pitch, unsigned int width, unsigned int height, unsigned int depth)
{
	// Try to determine the optimal block dimensions
	unsigned int dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
	unsigned int dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
	dim3 dimBlock(dimX, dimY);

	// Replace the voxel values by the b-spline coefficients
	dim3 dimGridX(height / dimBlock.x, depth / dimBlock.y);
	SamplesToCoefficients3DX<floatN><<<dimGridX, dimBlock>>>(volume, pitch, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DX kernel failed");

	dim3 dimGridY(width / dimBlock.x, depth / dimBlock.y);
	SamplesToCoefficients3DY<floatN><<<dimGridY, dimBlock>>>(volume, pitch, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DY kernel failed");

	dim3 dimGridZ(width / dimBlock.x, height / dimBlock.y);
	SamplesToCoefficients3DZ<floatN><<<dimGridZ, dimBlock>>>(volume, pitch, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DZ kernel failed");
}

//! Convert the voxel values into cubic b-spline coefficients
//! @param volume  pointer to the voxel volume in GPU (device) memory
//! @param pitch   width in bytes (including padding bytes)
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
//! @note Prints stopwatch feedback
template<typename floatN>
void CubicBSplinePrefilter3DTimer(floatN* volume, unsigned int pitch, unsigned int width, unsigned int height, unsigned int depth)
{
	printf("\nCubic B-Spline Prefilter timer:\n");
	unsigned int hTimer;
	CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	CUT_SAFE_CALL(cutResetTimer(hTimer));
	CUT_SAFE_CALL(cutStartTimer(hTimer));

	// Try to determine the optimal block dimensions
	unsigned int dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
	unsigned int dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
	dim3 dimBlock(dimX, dimY);

	// Replace the voxel values by the b-spline coefficients
	dim3 dimGridX(height / dimBlock.x, depth / dimBlock.y);
	SamplesToCoefficients3DX<floatN><<<dimGridX, dimBlock>>>(volume, pitch, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DX kernel failed");

	CUT_SAFE_CALL(cutStopTimer(hTimer));
	double timerValueX = cutGetTimerValue(hTimer);
	printf("x-direction : %f msec\n", timerValueX);
	CUT_SAFE_CALL(cutResetTimer(hTimer));
	CUT_SAFE_CALL(cutStartTimer(hTimer));

	dim3 dimGridY(width / dimBlock.x, depth / dimBlock.y);
	SamplesToCoefficients3DY<floatN><<<dimGridY, dimBlock>>>(volume, pitch, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DY kernel failed");

	CUT_SAFE_CALL(cutStopTimer(hTimer));
	double timerValueY = cutGetTimerValue(hTimer);
	printf("y-direction : %f msec\n", timerValueY);
	CUT_SAFE_CALL(cutResetTimer(hTimer));
	CUT_SAFE_CALL(cutStartTimer(hTimer));

	dim3 dimGridZ(width / dimBlock.x, height / dimBlock.y);
	SamplesToCoefficients3DZ<floatN><<<dimGridZ, dimBlock>>>(volume, pitch, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DZ kernel failed");

	CUT_SAFE_CALL(cutStopTimer(hTimer));
	double timerValueZ = cutGetTimerValue(hTimer);
	printf("z-direction : %f msec\n", timerValueZ);
	printf("total : %f msec\n\n", timerValueX+timerValueY+timerValueZ);
}