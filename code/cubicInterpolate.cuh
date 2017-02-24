#pragma once

#include <vector>
#include <vector_types.h>

std::vector<float> Interpolate2D(cudaTextureObject_t tex, const std::vector<float2> & coords);