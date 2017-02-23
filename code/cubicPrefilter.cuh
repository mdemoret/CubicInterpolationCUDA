#pragma once

#include "Prefix.h"

//! Convert the pixel values into cubic b-spline coefficients
//! @param image  pointer to the image bitmap in GPU (device) memory
//! @param pitch   width in bytes (including padding bytes)
//! @param width   image width in number of pixels
//! @param height  image height in number of pixels
template<typename floatN>
void CUBIC_API CubicBSplinePrefilter2D(floatN* image, unsigned int pitch, unsigned int width, unsigned int height);

//! Convert the pixel values into cubic b-spline coefficients
//! @param image  pointer to the image bitmap in GPU (device) memory
//! @param pitch   width in bytes (including padding bytes)
//! @param width   image width in number of pixels
//! @param height  image height in number of pixels
//! @note Prints stopwatch feedback
template<typename floatN>
void CUBIC_API CubicBSplinePrefilter2DTimer(floatN* image, unsigned int pitch, unsigned int width, unsigned int height);

//! Convert the voxel values into cubic b-spline coefficients
//! @param volume  pointer to the voxel volume in GPU (device) memory
//! @param pitch   width in bytes (including padding bytes)
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
template<typename floatN>
void CUBIC_API CubicBSplinePrefilter3D(floatN* volume, unsigned int pitch, unsigned int width, unsigned int height, unsigned int depth);

//! Convert the voxel values into cubic b-spline coefficients
//! @param volume  pointer to the voxel volume in GPU (device) memory
//! @param pitch   width in bytes (including padding bytes)
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
//! @note Prints stopwatch feedback
template<typename floatN>
void CUBIC_API CubicBSplinePrefilter3DTimer(floatN* volume, unsigned int pitch, unsigned int width, unsigned int height, unsigned int depth);