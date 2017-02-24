#pragma once

#include "internal/bspline_kernel.cuh"
#include <texture_fetch_functions.h>

//! Linearly interpolated texture lookup, using unnormalized coordinates.
//! This function merely serves as a reference for the cubic versions.
//! @param tex  1D texture
//! @param x  unnormalized x texture coordinate
__device__ float linearTex1D(cudaTextureObject_t tex, float x)
{
   return tex1D<float>(tex, x);
}

//! Cubic interpolated texture lookup, using unnormalized coordinates.
//! Straight forward implementation, using 4 nearest neighbour lookups.
//! @param tex  1D texture
//! @param x  unnormalized x texture coordinate
__device__ float cubicTex1DSimple(cudaTextureObject_t tex, float x)
{
   // transform the coordinate from [0,extent] to [-0.5, extent-0.5]
   const float coord_grid = x - 0.5f;
   float index = floor(coord_grid);
   const float fraction = coord_grid - index;
   index += 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

   float result = 0.0f;
   for (float x=-1; x < 2.5f; x++)
   {
      float bsplineX = bspline(x-fraction);
      float u = index + x;
      result += bsplineX * tex1D<float>(tex, u);
   }
   return result;
}

//! Cubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 2 linear lookups.
//! @param tex  1D texture
//! @param x  unnormalized x texture coordinate
#define WEIGHTS bspline_weights
#define CUBICTEX1D cubicTex1D
#include "internal/cubicTex1D_kernel.inl"
#undef CUBICTEX1D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative
#define CUBICTEX1D cubicTex1D_1st_derivative
#include "internal/cubicTex1D_kernel.inl"
#undef CUBICTEX1D
#undef WEIGHTS


#ifdef cudaTextureType1DLayered
// support for layered texture calls
#undef _EXTRA_ARGS
#undef _PASS_EXTRA_ARGS
#undef _TEX1D
#define _EXTRA_ARGS , int layer
#define _PASS_EXTRA_ARGS , layer
#define _TEX1D tex1DLayered

#define WEIGHTS bspline_weights
#define CUBICTEX1D cubicTex1DLayered
#include "internal/cubicTex1D_kernel.inl"
#undef CUBICTEX1D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative
#define CUBICTEX1D cubicTex1DLayered_1st_derivative
#include "internal/cubicTex1D_kernel.inl"
#undef CUBICTEX1D
#undef WEIGHTS
#endif // cudaTextureType1DLayered

#undef _EXTRA_ARGS
#undef _PASS_EXTRA_ARGS
#undef _TEX1D
