/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2013, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.

When using this code in a scientific project, please cite one or all of the
following papers:
*  Daniel Ruijters and Philippe Th�venaz,
   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
   The Computer Journal, vol. 55, no. 1, pp. 15-20, January 2012.
   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
*  Daniel Ruijters, Bart M. ter Haar Romeny, and Paul Suetens,
   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
   Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.
\*--------------------------------------------------------------------------*/

#include <texture_fetch_functions.h>
#include "bspline_kernel.cu"

#ifndef _EXTRA_ARGS
#define _EXTRA_ARGS
#define _PASS_EXTRA_ARGS
#endif

#ifndef _TEX1D
#define _TEX1D tex1D
#endif

#ifndef WEIGHTS
#define WEIGHTS bspline_weights
#endif

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 2 linear lookups.
//! @param tex  1D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
template<typename floatN>
__device__ floatN CUBICTEX1D(cudaTextureObject_t tex, float x _EXTRA_ARGS)
{
   // transform the coordinate from [0,extent] to [-0.5, extent-0.5]
   const float coord_grid = x - 0.5f;
   const float index = floor(coord_grid);
   const float fraction = coord_grid - index;
   float w0, w1, w2, w3;
   WEIGHTS(fraction, w0, w1, w2, w3);

   const float g0 = w0 + w1;
   const float g1 = w2 + w3;
   const float h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
   const float h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

   // fetch the two linear interpolations
   floatN tex0 = _TEX1D<floatN>(tex, h0 _PASS_EXTRA_ARGS);
   floatN tex1 = _TEX1D<floatN>(tex, h1 _PASS_EXTRA_ARGS);

   // weigh along the x-direction
   return (g0 * tex0 + g1 * tex1);
}


// Specializations

// These specializations fill in the floatN and T class types and therefore
// allow the cubicTex1D/cubicTex1DLayered function to be called without any
// template arguments, thus without any <> brackets.

template __device__ float CUBICTEX1D<float>(cudaTextureObject_t tex, float x _EXTRA_ARGS);
template __device__ float2 CUBICTEX1D<float2>(cudaTextureObject_t tex, float x _EXTRA_ARGS);
template __device__ float4 CUBICTEX1D<float4>(cudaTextureObject_t tex, float x _EXTRA_ARGS);