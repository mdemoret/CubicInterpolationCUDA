#include "cubicTex.cuh"

#include "internal/bspline_kernel.cuh"
#include <texture_fetch_functions.h>
#include "helper_math.h"

//! Bilinearly interpolated texture lookup, using unnormalized coordinates.
//! This function merely serves as a reference for the bicubic versions.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
__device__ float linearTex2D(cudaTextureObject_t tex, float x, float y)
{
   return tex2D<float>(tex, x, y);
}

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Straight forward implementation, using 16 nearest neighbour lookups.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
__device__ float cubicTex2DSimple(cudaTextureObject_t tex, float x, float y)
{
   // transform the coordinate from [0,extent] to [-0.5, extent-0.5]
   const float2 coord_grid = make_float2(x - 0.5f, y - 0.5f);
   float2 index = floorf(coord_grid);
   const float2 fraction = coord_grid - index;
   index.x += 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]
   index.y += 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

   float result = 0.0f;
   for (float y=-1; y < 2.5f; y++)
   {
      float bsplineY = bspline(y-fraction.y);
      float v = index.y + y;
      for (float x=-1; x < 2.5f; x++)
      {
         float bsplineXY = bspline(x-fraction.x) * bsplineY;
         float u = index.x + x;
         result += bsplineXY * tex2D<float>(tex, u, v);
      }
   }
   return result;
}

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 4 trilinear lookups.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
#define WEIGHTS bspline_weights
#define CUBICTEX2D cubicTex2D
#include "internal/cubicTex2D_kernel.inl"
#undef CUBICTEX2D
#undef WEIGHTS

// Fast bicubic interpolated 1st order derivative texture lookup in x- and
// y-direction, using unnormalized coordinates.
__device__ void bspline_weights_1st_derivative_x(float2 fraction, float2& w0, float2& w1, float2& w2, float2& w3)
{
   float t0, t1, t2, t3;
   bspline_weights_1st_derivative(fraction.x, t0, t1, t2, t3);
   w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
   bspline_weights(fraction.y, t0, t1, t2, t3);
   w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
}

__device__ void bspline_weights_1st_derivative_y(float2 fraction, float2& w0, float2& w1, float2& w2, float2& w3)
{
   float t0, t1, t2, t3;
   bspline_weights(fraction.x, t0, t1, t2, t3);
   w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
   bspline_weights_1st_derivative(fraction.y, t0, t1, t2, t3);
   w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
}

#define WEIGHTS bspline_weights_1st_derivative_x
#define CUBICTEX2D cubicTex2D_1st_derivative_x
#include "internal/cubicTex2D_kernel.inl"
#undef CUBICTEX2D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative_y
#define CUBICTEX2D cubicTex2D_1st_derivative_y
#include "internal/cubicTex2D_kernel.inl"
#undef CUBICTEX2D
#undef WEIGHTS


#ifdef cudaTextureType2DLayered
// support for layered texture calls
#undef _EXTRA_ARGS
#undef _PASS_EXTRA_ARGS
#undef _TEX2D
#define _EXTRA_ARGS , int layer
#define _PASS_EXTRA_ARGS , layer
#define _TEX2D tex2DLayered

#define WEIGHTS bspline_weights
#define CUBICTEX2D cubicTex2DLayered
#include "internal/cubicTex2D_kernel.inl"
#undef CUBICTEX2D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative_x
#define CUBICTEX2D cubicTex2DLayered_1st_derivative_x
#include "internal/cubicTex2D_kernel.inl"
#undef CUBICTEX2D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative_y
#define CUBICTEX2D cubicTex2DLayered_1st_derivative_y
#include "internal/cubicTex2D_kernel.inl"
#undef CUBICTEX2D
#undef WEIGHTS
#endif //cudaTextureType2DLayered

#undef _EXTRA_ARGS
#undef _PASS_EXTRA_ARGS
#undef _TEX2D
