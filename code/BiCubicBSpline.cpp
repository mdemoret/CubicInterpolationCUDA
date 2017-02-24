#include "BiCubicBSpline.hpp"
#include "cuda_runtime.h"
#include "memcpy.cuh"
#include "cubicPrefilter.cuh"
#include "cutil.h"
#include "cuda_texture_types.h"
#include "cubicInterpolate.cuh"

BiCubicBSpline::BiCubicBSpline()
{
}


BiCubicBSpline::~BiCubicBSpline()
{
}

void BiCubicBSpline::LoadInputData(const float * data, unsigned int width, unsigned int height)
{
   m_Width = width;
   m_Height = height;

   // calculate the b-spline coefficients
   cudaPitchedPtr bsplineCoeffs = CubicBSplinePrefilter2D(data, 0, width, height);

   // Create the B-spline coefficients texture
   cudaChannelFormatDesc channelDescCoeff = cudaCreateChannelDesc<float>();
   cudaArray *coeffArray = 0;
   CUDA_SAFE_CALL(cudaMallocArray(&coeffArray, &channelDescCoeff, width, height));
   CUDA_SAFE_CALL(cudaMemcpy2DToArray(coeffArray, 0, 0, bsplineCoeffs.ptr, bsplineCoeffs.pitch, width * sizeof(float), height, cudaMemcpyDeviceToDevice));

   cudaResourceDesc resDesc;
   memset(&resDesc, 0, sizeof(resDesc));
   resDesc.resType = cudaResourceTypeArray;
   resDesc.res.array.array = coeffArray;

   cudaTextureDesc texDesc;
   memset(&texDesc, 0, sizeof(texDesc));
   texDesc.addressMode[0] = cudaAddressModeWrap;
   texDesc.addressMode[1] = cudaAddressModeClamp;
   texDesc.filterMode = cudaFilterModeLinear;
   texDesc.readMode = cudaReadModeElementType;
   texDesc.normalizedCoords = 1;

   CUDA_SAFE_CALL(cudaCreateTextureObject(&m_Tex, &resDesc, &texDesc, NULL));
}


std::vector<float> BiCubicBSpline::Interpolate(const std::vector<float> & xCoords, const std::vector<float> & yCoords) const
{
   assert(xCoords.size() == yCoords.size());

   //loop and build the float2
   std::vector<float2> coords(xCoords.size());
   
   for (size_t i = 0; i < xCoords.size(); i++)
   {
      coords[i] = make_float2(xCoords[i], yCoords[i]);
   }

   return Interpolate2D(m_Tex, coords);
}