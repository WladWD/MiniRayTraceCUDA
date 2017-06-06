#include "Engine.h"




TraceEngine::Engine::Engine(int32_t mWidth, int32_t mHeight)
{
	mCamera = new Camera::Camera(1.0f, 1000.0, mWidth, mHeight, 1.0f);
	mDrawTrace = new DrawTrace::DrawTrace(mCamera, mWidth, mHeight);
}

TraceEngine::Engine::~Engine()
{
	delete mCamera;
	delete mDrawTrace;
}

void TraceEngine::Engine::Tick(void)
{
}

void TraceEngine::Engine::Draw()
{
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	mDrawTrace->Draw();
}

void TraceEngine::Engine::Resize(int32_t mWidth, int32_t mHeight)
{
	mCamera->Resize(mWidth, mHeight);
	mDrawTrace->Resize(mWidth, mHeight);
}
