#pragma once

#ifdef CUBIC_EXPORTS
#define CUBIC_API __declspec(dllexport)
#else
#define CUBIC_API __declspec(dllimport)
#endif