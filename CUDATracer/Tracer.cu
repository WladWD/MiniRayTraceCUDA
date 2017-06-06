#include "Tracer.cuh"


__global__ void GradientCompute(float4 *mTextureBuffer, int32_t mDimensionX, int32_t mDimensionY)
{
	int gDispatchX = blockIdx.x * blockDim.x + threadIdx.x;
	int mDispatchY = blockIdx.y * blockDim.y + threadIdx.y;

	if (gDispatchX < mDimensionX && mDispatchY < mDimensionY)
	{
		mTextureBuffer[mDispatchY * mDimensionX + gDispatchX] =
			make_float4((float)gDispatchX / mDimensionX, (float)mDispatchY / mDimensionY, 1.0f, 1.0f);
	}
}


TracerCUDA::Tracer::Tracer(const TraceInfo &mInfo) : mInfo(mInfo)
{
	cudaDeviceProp cudaDevice_prop;
	memset(&cudaDevice_prop, 0, sizeof(cudaDevice_prop));
	cudaDevice_prop.major = 1;
	cudaDevice_prop.minor = 3;

	CUDA_HR(cudaChooseDevice(&cudaGLDevice_ID, &cudaDevice_prop));
	CUDA_HR(cudaGLSetGLDevice(cudaGLDevice_ID));
}

TracerCUDA::Tracer::~Tracer()
{
	CUDA_HR(cudaGraphicsUnregisterResource(mCUDA_texture));
}

void TracerCUDA::Tracer::Resize(const TraceInfo &mInfo)
{
	this->mInfo = mInfo;
}

void TracerCUDA::Tracer::InitTexture(void)
{
	CUDA_HR(cudaGraphicsGLRegisterBuffer(&mCUDA_texture, mInfo.mTextureBuffer, cudaGraphicsMapFlagsNone));
}

void TracerCUDA::Tracer::MapResource(void)
{
	CUDA_HR(cudaGraphicsMapResources(1, &mCUDA_texture, 0));

	size_t mSize = mInfo.mSizeTextureBuffer;
	CUDA_HR(cudaGraphicsResourceGetMappedPointer((void **)&mCUDA_BUFFER, &mSize, mCUDA_texture));
}

void TracerCUDA::Tracer::UnmapResource(void)
{
	CUDA_HR(cudaDeviceSynchronize());

	CUDA_HR(cudaGraphicsUnmapResources(1, &mCUDA_texture, 0));
}

void TracerCUDA::Tracer::Trace(void)
{
	dim3 threads(32, 32);
	dim3 blocks((mInfo.mDimX + 31) / 32, (mInfo.mDimY + 31) / 32);
	GradientCompute <<<blocks, threads >>> (mCUDA_BUFFER, mInfo.mDimX, mInfo.mDimY);
}