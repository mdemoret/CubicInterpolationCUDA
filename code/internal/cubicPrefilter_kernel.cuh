#pragma once

#include <cuda_runtime.h>

template<typename floatN>
__host__ __device__ void ConvertToInterpolationCoefficients(
   floatN* coeffs,		// input samples --> output coefficients
   unsigned int DataLength,	// number of samples or coefficients
   int step);			// element interleave in bytes