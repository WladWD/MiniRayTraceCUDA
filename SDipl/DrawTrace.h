#include "MainHeder.h"
#include "Camera.h"
//#define MyImpotr
#include "Tracer.cuh"

#pragma once
namespace DrawTrace
{
	class DrawTrace
	{
		int32_t mWidth;
		int32_t mHeight;
		//////////////////////////////////////////////////////
		GLuint mTexture;
		GLuint mTextureBuffer;
		//////////////////////////////////////////////////////
		GLuint mFramebuffer;
		//////////////////////////////////////////////////////
		Camera::Camera *mCamera;
		TracerCUDA::Tracer *mTrace;
		//////////////////////////////////////////////////////
		void Init(void);
	public:
		DrawTrace(Camera::Camera *mCamera, int32_t mWidth, int32_t mHeight);
		~DrawTrace();
		//////////////////////////////////////////////////////
		void Resize(int32_t mWidth, int32_t mHeight);
		void Draw(void);
	};
}
