#include "DrawTrace.h"


DrawTrace::DrawTrace::DrawTrace(Camera::Camera *mCamera, int32_t mWidth, int32_t mHeight) : mCamera(mCamera), mWidth(mWidth), mHeight(mHeight)
{
	Init();

	TracerCUDA::TraceInfo mTraceInfo = { mTextureBuffer, mWidth * mHeight * sizeof(vec4), mWidth, mHeight };
	mTrace = new TracerCUDA::Tracer(mTraceInfo);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mTextureBuffer);
	mTrace->InitTexture();
}

DrawTrace::DrawTrace::~DrawTrace()
{
	delete mTrace;
	glDeleteTextures(1, &mTexture);
	glDeleteBuffers(1, &mTextureBuffer);
	glDeleteFramebuffers(1, &mFramebuffer);
}

void DrawTrace::DrawTrace::Init(void)
{
	//////////////////////////////////////////////////
	glGenBuffers(1, &mTextureBuffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mTextureBuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, mWidth * mHeight * sizeof(vec4), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	//////////////////////////////////////////////////
	glGenTextures(1, &mTexture);
	glBindTexture(GL_TEXTURE_2D, mTexture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mWidth,  mHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glBindTexture(GL_TEXTURE_2D, 0);
	//////////////////////////////////////////////////
	glGenFramebuffers(1, &mFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, mFramebuffer);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,
		GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mTexture, 0);
	//////////////////////////////////////////////////
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void DrawTrace::DrawTrace::Resize(int32_t mWidth, int32_t mHeight)
{
	glDeleteTextures(1, &mTexture);
	glDeleteBuffers(1, &mTextureBuffer);
	glDeleteFramebuffers(1, &mFramebuffer);

	this->mWidth = mWidth;
	this->mHeight = mHeight;
	Init();

	TracerCUDA::TraceInfo mTraceInfo = { mTextureBuffer, mWidth * mHeight * sizeof(vec4), mWidth, mHeight };
	mTrace->Resize(mTraceInfo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mTextureBuffer);
	mTrace->InitTexture();
}

void DrawTrace::DrawTrace::Draw(void)
{
	glm::vec3 mPosition = mCamera->GetPosition();
	glm::mat4 mWorld = glm::transpose(glm::inverse(mCamera->GetProjViewMatrix()));
	////////////////////////////////////////////////////////////////////////////////////////////////////
	float mMatrix[16];
	memcpy(mMatrix + 0,		glm::value_ptr(mWorld[0]), sizeof(glm::vec4));
	memcpy(mMatrix + 4,		glm::value_ptr(mWorld[1]), sizeof(glm::vec4));
	memcpy(mMatrix + 8,		glm::value_ptr(mWorld[2]), sizeof(glm::vec4));
	memcpy(mMatrix + 12,	glm::value_ptr(mWorld[3]), sizeof(glm::vec4));
	////////////////////////////////////////////////////////////////////////////////////////////////////
	mTrace->MapResource();
	mTrace->Trace(glm::value_ptr(mPosition), mMatrix);
	mTrace->UnmapResource();
	glBindTexture(GL_TEXTURE_2D, mTexture);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mTextureBuffer);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGBA, GL_FLOAT, 0);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	//glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mFramebuffer);
	//glClearColor(0.0f, 1.0f, 0.0f, 1.0);
	//glClear(GL_COLOR_BUFFER_BIT);




	glBindFramebuffer(GL_READ_FRAMEBUFFER, mFramebuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	/*glBindFramebuffer(GL_READ_FRAMEBUFFER, renderContext->multiSamplingContext.fbo);
glBindFramebuffer(GL_DRAW_FRAMEBUFFER, renderContext->beforeEffectsContext.fbo);*/
	glBlitFramebuffer(0, 0, mWidth, mHeight, 0, 0, mWidth, mHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
}