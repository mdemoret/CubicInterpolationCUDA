#pragma once

#include <cuda_runtime.h>
#include "version.cuh"

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef signed char schar;

inline __device__ __host__ unsigned int UMIN(unsigned int a, unsigned int b)
{
   return a < b ? a : b;
}

inline __device__ __host__ unsigned int PowTwoDivider(unsigned int n)
{
   if (n == 0) return 0;
   unsigned int divider = 1;
   while ((n & divider) == 0) divider <<= 1; 
   return divider;
}

inline __host__ __device__ float2 operator-(float a, float2 b)
{
   return make_float2(a - b.x, a - b.y);
}

inline __host__ __device__ float3 operator-(float a, float3 b)
{
   return make_float3(a - b.x, a - b.y, a - b.z);
}