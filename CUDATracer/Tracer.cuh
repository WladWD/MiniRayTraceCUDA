#include "TraceInfo.cuh"

#pragma once
namespace TracerCUDA
{
	extern "C" class Tracer
	{
		int cudaGLDevice_ID;
		TraceInfo mInfo;

		cudaGraphicsResource *mCUDA_texture;
		float4 *mCUDA_BUFFER;
	public:
		MyLibrary Tracer(const TraceInfo &mInfo);
		MyLibrary ~Tracer();

		MyLibrary void Resize(const TraceInfo &mInfo);
		MyLibrary void InitTexture(void);//need call bindtexture beforce call this method
		MyLibrary void MapResource(void);
		MyLibrary void UnmapResource(void);

		MyLibrary void Trace(void);
	};
}

