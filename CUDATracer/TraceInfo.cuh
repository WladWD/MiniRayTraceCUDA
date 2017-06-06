#include <glew.h>
#include <Windows.h>
#include <cuda.h>
#include "cuda_runtime_api.h"
#include <cuda_gl_interop.h>
#include <stdint.h>

#pragma once
namespace TracerCUDA
{
	struct TraceInfo
	{
		//OpenGL resource
		GLuint mTextureBuffer;
		int32_t mSizeTextureBuffer;
		int32_t mDimX;
		int32_t mDimY;
	};
}

#ifndef MyImpotr
#define MyLibrary __declspec(dllexport)
#else
#define MyLibrary //__declspec(dllimport)
#endif

#if defined(DEBUG) | defined(_DEBUG)
#ifndef CUDA_HR
#define CUDA_HR(x)																					\
{																									\
cudaError_t cudaStatusError = x;																	\
	if(cudaStatusError != cudaSuccess)																\
	{																								\
		MessageBoxA(NULL, "CUDA Error", cudaGetErrorString(cudaStatusError), MB_OK || MB_ICONERROR);\
		throw L"CUDA Error";																		\
	}																								\
}
#endif
#else
#ifndef CUDA_HR
#define CUDA_HR(x) (x)
#endif
#endif