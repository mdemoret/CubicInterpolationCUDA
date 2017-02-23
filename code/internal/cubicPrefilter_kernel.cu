#include "cubicPrefilter_kernel.cuh"
#include "math_func.cuh"

// The code below is based on the work of Philippe Thevenaz.
// See <http://bigwww.epfl.ch/thevenaz/interpolation/>

#define Pole (sqrt(3.0f)-2.0f)  //pole for cubic b-spline
 
//--------------------------------------------------------------------------
// Local GPU device procedures
//--------------------------------------------------------------------------
template<typename floatN>
__host__ __device__ floatN InitialCausalCoefficient(
	floatN* c,			// coefficients
	unsigned int DataLength,	// number of coefficients
	int step)			// element interleave in bytes
{
	const unsigned int Horizon = UMIN(12, DataLength);

	// this initialization corresponds to clamping boundaries
	// accelerated loop
	float zn = Pole;
	floatN Sum = *c;
	for (unsigned int n = 0; n < Horizon; n++) {
		Sum += zn * *c;
		zn *= Pole;
		c = (floatN*)((unsigned char*)c + step);
	}
	return(Sum);
}

template<typename floatN>
__host__ __device__ floatN InitialAntiCausalCoefficient(
	floatN* c,			// last coefficient
	unsigned int DataLength,	// number of samples or coefficients
	int step)			// element interleave in bytes
{
	// this initialization corresponds to clamping boundaries
	return((Pole / (Pole - 1.0f)) * *c);
}

template<typename floatN>
__host__ __device__ void ConvertToInterpolationCoefficients(
	floatN* coeffs,		// input samples --> output coefficients
	unsigned int DataLength,	// number of samples or coefficients
	int step)			// element interleave in bytes
{
	// compute the overall gain
	const float Lambda = (1.0f - Pole) * (1.0f - 1.0f / Pole);

	// causal initialization
	floatN* c = coeffs;
	floatN previous_c;  //cache the previously calculated c rather than look it up again (faster!)
	*c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, step);
	// causal recursion
	for (unsigned int n = 1; n < DataLength; n++) {
		c = (floatN*)((unsigned char*)c + step);
		*c = previous_c = Lambda * *c + Pole * previous_c;
	}
	// anticausal initialization
	*c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
	// anticausal recursion
	for (int n = DataLength - 2; 0 <= n; n--) {
		c = (floatN*)((unsigned char*)c - step);
		*c = previous_c = Pole * (previous_c - *c);
	}
}

template void ConvertToInterpolationCoefficients<float>(float* coeffs, unsigned int DataLength, int step);