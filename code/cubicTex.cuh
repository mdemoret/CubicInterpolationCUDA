#pragma once

#include "cuda_runtime.h"

#ifndef _EXTRA_ARGS
#define _EXTRA_ARGS
#define _PASS_EXTRA_ARGS
#endif

#ifndef _TEX1D
#define _TEX1D tex1D
#endif

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 2 linear lookups.
//! @param tex  1D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
template<typename floatN>
__device__ floatN cubicTex1D(cudaTextureObject_t tex, float x);

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 2 linear lookups.
//! @param tex  1D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
template<typename floatN>
__device__ floatN cubicTex1D_1st_derivative(cudaTextureObject_t tex, float x);


#ifndef _TEX2D
#define _TEX2D tex2D
#endif

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 4 bilinear lookups.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
template<typename floatN>
__device__ floatN cubicTex2D(cudaTextureObject_t tex, float x, float y _EXTRA_ARGS);

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 4 bilinear lookups.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
template<typename floatN>
__device__ floatN cubicTex2D_1st_derivative_x(cudaTextureObject_t tex, float x, float y _EXTRA_ARGS);

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 4 bilinear lookups.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
template<typename floatN>
__device__ floatN cubicTex2D_1st_derivative_y(cudaTextureObject_t tex, float x, float y _EXTRA_ARGS);

#ifndef _TEX3D
#define _TEX3D tex3D
#endif

//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 8 trilinear lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<typename floatN>
__device__ floatN cubicTex3D(cudaTextureObject_t tex, float3 coord);

//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 8 trilinear lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<typename floatN>
__device__ floatN cubicTex3D_1st_derivative_x(cudaTextureObject_t tex, float3 coord);

//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 8 trilinear lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<typename floatN>
__device__ floatN bspline_weights_1st_derivative_y(cudaTextureObject_t tex, float3 coord);

//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 8 trilinear lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<typename floatN>
__device__ floatN bspline_weights_1st_derivative_z(cudaTextureObject_t tex, float3 coord);