#include "cubicPrefilter.cuh"

#include <stdio.h>
#include "cutil.h"
#include "internal/cubicPrefilter_kernel.cuh"
#include "internal/math_func.cuh"

// ***************************************************************************
// *	Global GPU procedures
// ***************************************************************************
template<typename floatN>
__global__ void SamplesToCoefficients2DX(
   floatN* image,		// in-place processing
   unsigned int pitch,			// width in bytes
   unsigned int width,			// width of the image
   unsigned int height)		// height of the image
{
   // process lines in x-direction
   const unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
   floatN* line = (floatN*)((unsigned char*)image + y * pitch);  //direct access

   ConvertToInterpolationCoefficients(line, width, sizeof(floatN));
}

template<typename floatN>
__global__ void SamplesToCoefficients2DY(
   floatN* image,		// in-place processing
   unsigned int pitch,			// width in bytes
   unsigned int width,			// width of the image
   unsigned int height)		// height of the image
{
   // process lines in x-direction
   const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
   floatN* line = image + x;  //direct access

   ConvertToInterpolationCoefficients(line, height, pitch);
}

// ***************************************************************************
// *	Exported functions
// ***************************************************************************

//! Convert the pixel values into cubic b-spline coefficients
//! @param image  pointer to the image bitmap in GPU (device) memory
//! @param pitch   width in bytes (including padding bytes)
//! @param width   image width in number of pixels
//! @param height  image height in number of pixels
template<typename floatN>
cudaPitchedPtr CubicBSplinePrefilter2D(floatN* image, unsigned int pitch, unsigned int width, unsigned int height)
{
   cudaPitchedPtr imageData;
   CUDA_SAFE_CALL(cudaMalloc3D(&imageData, make_cudaExtent(width, height, 1)));

   CUDA_SAFE_CALL(cudaMemcpy2D(imageData.ptr, imageData.pitch, image, sizeof(floatN) * width, sizeof(floatN) * width, height, cudaMemcpyHostToDevice));

   dim3 dimBlockX(min(PowTwoDivider(height), 64));
   dim3 dimGridX(height / dimBlockX.x);
   SamplesToCoefficients2DX<floatN><<<dimGridX, dimBlockX>>>(image, pitch, width, height);
   CUT_CHECK_ERROR("SamplesToCoefficients2DX kernel failed");

   dim3 dimBlockY(min(PowTwoDivider(width), 64));
   dim3 dimGridY(width / dimBlockY.x);
   SamplesToCoefficients2DY<floatN><<<dimGridY, dimBlockY>>>(image, pitch, width, height);
   CUT_CHECK_ERROR("SamplesToCoefficients2DY kernel failed");

   return imageData;
}

template cudaPitchedPtr CubicBSplinePrefilter2D<float>(float* image, unsigned int pitch, unsigned int width, unsigned int height);

//! Convert the pixel values into cubic b-spline coefficients
//! @param image  pointer to the image bitmap in GPU (device) memory
//! @param pitch   width in bytes (including padding bytes)
//! @param width   image width in number of pixels
//! @param height  image height in number of pixels
//! @note Prints stopwatch feedback
template<typename floatN>
void CubicBSplinePrefilter2DTimer(floatN* image, unsigned int pitch, unsigned int width, unsigned int height)
{
   printf("\nCubic B-Spline Prefilter timer:\n");
   unsigned int hTimer;
   CUT_SAFE_CALL(cutCreateTimer(&hTimer));
   CUT_SAFE_CALL(cutResetTimer(hTimer));
   CUT_SAFE_CALL(cutStartTimer(hTimer));

   dim3 dimBlockX(min(PowTwoDivider(height), 64));
   dim3 dimGridX(height / dimBlockX.x);
   SamplesToCoefficients2DX<floatN><<<dimGridX, dimBlockX>>>(image, pitch, width, height);
   CUT_CHECK_ERROR("SamplesToCoefficients2DX kernel failed");

   CUT_SAFE_CALL(cutStopTimer(hTimer));
   double timerValueX = cutGetTimerValue(hTimer);
   printf("x-direction : %f msec\n", timerValueX);
   CUT_SAFE_CALL(cutResetTimer(hTimer));
   CUT_SAFE_CALL(cutStartTimer(hTimer));

   dim3 dimBlockY(min(PowTwoDivider(width), 64));
   dim3 dimGridY(width / dimBlockY.x);
   SamplesToCoefficients2DY<floatN><<<dimGridY, dimBlockY>>>(image, pitch, width, height);
   CUT_CHECK_ERROR("SamplesToCoefficients2DY kernel failed");

   CUT_SAFE_CALL(cutStopTimer(hTimer));
   double timerValueY = cutGetTimerValue(hTimer);
   printf("y-direction : %f msec\n", timerValueY);
   printf("total : %f msec\n\n", timerValueX+timerValueY);
}