//#pragma once
//__pragma(warning(push))
//__pragma(warning(disable:4201))
//
//#include <glm\glm.hpp>
//#include <glm\gtx\compatibility.hpp>
//
//__pragma(warning(pop))

#include "Prefix.h"
#include <memory>

typedef unsigned long long cudaTextureObject_t;

class CUBIC_API BiCubicBSpline
{
public:
   BiCubicBSpline();
   ~BiCubicBSpline();

   void LoadInputData(const float * data, unsigned int width, unsigned int height);

   float Interpolate(float x, float y) const;

private:
   unsigned int m_Width;
   unsigned int m_Height;

   cudaTextureObject_t m_Tex;
};

