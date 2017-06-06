#include "DrawTrace.h"

#pragma once
namespace TraceEngine
{
	class Engine
	{
		Camera::Camera *mCamera;
		DrawTrace::DrawTrace *mDrawTrace;
	public:
		Engine(int32_t mWidth, int32_t mHeight);
		~Engine();

		void Tick(void);
		void Draw();
		void Resize(int32_t mWidth, int32_t mHeight);
	};
}

