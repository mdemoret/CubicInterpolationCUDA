#include "cubicInterpolate.cuh"
#include "cutil.h"
#include "cuda_runtime.h"
#include "cubicTex2D.cuh"

__global__
void Interpolate2D_kernel(cudaTextureObject_t tex, float2 * inputCoords, size_t count, float* outputVals)
{
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx < count)
   {
      float2 coord = inputCoords[idx];

      outputVals[idx] = cubicTex2D<float>(tex, coord.x, coord.y);
   }
}

__global__
void InterpolateWithDeriv2D(cudaTextureObject_t tex, float2 * inputCoords, size_t count, float* outputVals, float2 * outputDerivs)
{
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx < count)
   {
      float2 coord = inputCoords[idx];

      outputVals[idx] = cubicTex2D<float>(tex, coord.x, coord.y);
      outputDerivs[idx].x = cubicTex2D_1st_derivative_x<float>(tex, coord.x, coord.y);
      outputDerivs[idx].y = cubicTex2D_1st_derivative_y<float>(tex, coord.x, coord.y);
   }
}

std::vector<float> Interpolate2D(cudaTextureObject_t tex, const std::vector<float2> & coords)
{
   float2 * inputCoords_D;
   CUDA_SAFE_CALL(cudaMalloc(&inputCoords_D, sizeof(float2) * coords.size()));

   CUDA_SAFE_CALL(cudaMemcpy(inputCoords_D, &coords[0], sizeof(float2) * coords.size(), cudaMemcpyHostToDevice));

   float * outputVals_D;
   CUDA_SAFE_CALL(cudaMalloc(&outputVals_D, sizeof(float) * coords.size()));

   const unsigned int blockDim = 128;

   unsigned int blockCount = (unsigned int)ceil(coords.size() / (double)blockDim);

   Interpolate2D_kernel << <blockCount, blockDim >> >(tex, inputCoords_D, coords.size(), outputVals_D);

   std::vector<float> output(coords.size());

   CUDA_SAFE_CALL(cudaMemcpy(&output[0], outputVals_D, sizeof(float) * coords.size(), cudaMemcpyDeviceToHost));

   CUDA_SAFE_CALL(cudaDeviceSynchronize());

   return output;
}