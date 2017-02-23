#include "cubicTex.cuh"

#include "internal/bspline_kernel.cu"

//! Trilinearly interpolated texture lookup, using unnormalized coordinates.
//! This function merely serves as a reference for the tricubic versions.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<typename T, enum cudaTextureReadMode mode>
__device__ float linearTex3D(texture<T, _TEXTYPE3D, mode> tex, float3 coord)
{
   return tex3D(tex, coord.x, coord.y, coord.z);
}

//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Straight forward implementation, using 64 nearest neighbour lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<typename T, enum cudaTextureReadMode mode>
__device__ float cubicTex3DSimple(texture<T, _TEXTYPE3D, mode> tex, float3 coord)
{
   // transform the coordinate from [0,extent] to [-0.5, extent-0.5]
   const float3 coord_grid = coord - 0.5f;
   float3 index = floor(coord_grid);
   const float3 fraction = coord_grid - index;
   index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

   float result = 0.0f;
   for (float z=-1; z < 2.5f; z++)  //range [-1, 2]
   {
      float bsplineZ = bspline(z-fraction.z);
      float w = index.z + z;
      for (float y=-1; y < 2.5f; y++)
      {
         float bsplineYZ = bspline(y-fraction.y) * bsplineZ;
         float v = index.y + y;
         for (float x=-1; x < 2.5f; x++)
         {
            float bsplineXYZ = bspline(x-fraction.x) * bsplineYZ;
            float u = index.x + x;
            result += bsplineXYZ * tex3D(tex, u, v, w);
         }
      }
   }
   return result;
}

//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 8 trilinear lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
#define WEIGHTS bspline_weights
#define CUBICTEX3D cubicTex3D
#include "internal/cubicTex3D_kernel.inl"
#undef CUBICTEX3D
#undef WEIGHTS

// Fast tricubic interpolated 1st order derivative texture lookup in x-, y-
// and z-direction, using unnormalized coordinates.
__device__ void bspline_weights_1st_derivative_x(float3 fraction, float3& w0, float3& w1, float3& w2, float3& w3)
{
   float t0, t1, t2, t3;
   bspline_weights_1st_derivative(fraction.x, t0, t1, t2, t3);
   w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
   bspline_weights(fraction.y, t0, t1, t2, t3);
   w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
   bspline_weights(fraction.z, t0, t1, t2, t3);
   w0.z = t0; w1.z = t1; w2.z = t2; w3.z = t3;
}

__device__ void bspline_weights_1st_derivative_y(float3 fraction, float3& w0, float3& w1, float3& w2, float3& w3)
{
   float t0, t1, t2, t3;
   bspline_weights(fraction.x, t0, t1, t2, t3);
   w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
   bspline_weights_1st_derivative(fraction.y, t0, t1, t2, t3);
   w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
   bspline_weights(fraction.z, t0, t1, t2, t3);
   w0.z = t0; w1.z = t1; w2.z = t2; w3.z = t3;
}

__device__ void bspline_weights_1st_derivative_z(float3 fraction, float3& w0, float3& w1, float3& w2, float3& w3)
{
   float t0, t1, t2, t3;
   bspline_weights(fraction.x, t0, t1, t2, t3);
   w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
   bspline_weights(fraction.y, t0, t1, t2, t3);
   w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
   bspline_weights_1st_derivative(fraction.z, t0, t1, t2, t3);
   w0.z = t0; w1.z = t1; w2.z = t2; w3.z = t3;
}

#define WEIGHTS bspline_weights_1st_derivative_x
#define CUBICTEX3D cubicTex3D_1st_derivative_x
#include "internal/cubicTex3D_kernel.inl"
#undef CUBICTEX3D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative_y
#define CUBICTEX3D cubicTex3D_1st_derivative_y
#include "internal/cubicTex3D_kernel.inl"
#undef CUBICTEX3D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative_z
#define CUBICTEX3D cubicTex3D_1st_derivative_z
#include "internal/cubicTex3D_kernel.inl"
#undef CUBICTEX3D
#undef WEIGHTS
