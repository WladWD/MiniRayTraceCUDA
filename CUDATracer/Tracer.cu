#include "Tracer.cuh"

#define EPS 1e-3
#define BACKGROUND make_float4(0.0f, 0.0f, 0.0f, 1.0f)

#define BASE_SPHERE make_float3(0.0f, 0.0f, -8.0f)
#define BASE_SPHERE_RADIUS 2.0f

__constant__ float4 mWorldPosition[1];
__constant__ float4 mWorldMatrix[4];

__device__ float VectorDot(float4 A, float4 B) 
{
	return A.x * B.x +
		A.y * B.y +
		A.z * B.z +
		A.w * B.w;
}

__device__ float VectorDot(float3 A, float3 B)
{
	return A.x * B.x +
		A.y * B.y +
		A.z * B.z;
}

__device__ float4 VectorSub(float4 A, float4 B) 
{
	return make_float4(A.x - B.x,
		A.y - B.y,
		A.z - B.z,
		A.w - B.w);
}

__device__ float3 VectorSub(float3 A, float3 B)
{
	return make_float3(
		A.x - B.x,
		A.y - B.y,
		A.z - B.z
		);
}

__device__ float3 NormalizeVector(float3 A) 
{
	float d = A.x * A.x + A.y * A.y + A.z * A.z;
	d = sqrt(d);
	return make_float3(
		A.x / d,
		A.y / d,
		A.z / d
	);
}

__device__ float4 BuildDirection(float4 mSource) 
{
	float4 mDestDirection = make_float4(
		VectorDot(mWorldMatrix[0], mSource),
		VectorDot(mWorldMatrix[1], mSource),
		VectorDot(mWorldMatrix[2], mSource),
		VectorDot(mWorldMatrix[3], mSource));

	mDestDirection.x /= mDestDirection.w;
	mDestDirection.y /= mDestDirection.w;
	mDestDirection.z /= mDestDirection.w;
	mDestDirection.w = 0.0f;

	return mDestDirection;
}

__device__ float SolveQU(float a, float b, float c) 
{
	float tx = -c / b;
	if (a == 0.0f)
		return tx >= 1.0f ? tx : 0.0f;

	float D = b * b - 4.0f * a * c;

	if (D < 0.0f)
		return 0.0f;

	float DSQ = sqrt(D);
	float x1 = (-b + DSQ) / (2.0f * a);
	float x2 = (-b - DSQ) / (2.0f * a);
	
	x1 = x1 >= 1.0f ? x1 : 0.0f;
	x2 = x2 >= 1.0f ? x2 : 0.0f;
	return min(x1, x2);
}

__device__ float3 MakePointFromLine(float4 mRayStart, float4 mRayDir, float mT) 
{
	return make_float3(
		mRayStart.x + mRayDir.x * mT,
		mRayStart.y + mRayDir.y * mT,
		mRayStart.z + mRayDir.z * mT
	);
}

__device__ float4 IntersectRaySphere(float4 mRayStart, float4 mRayDir)
{

	float3 mSphere = BASE_SPHERE;
	float3 mSub = make_float3(
		mRayStart.x - mSphere.x,
		mRayStart.y - mSphere.y,
		mRayStart.z - mSphere.z
	);

	float a, b, c;
	a = VectorDot(mRayDir, mRayDir);//mRayDir.x * mRayDir.x + mRayDir.y *mRayDir.y + mRayDir.z * mRayDir.z;
	b = 2.0f * VectorDot(
		make_float3(mRayDir.x, mRayDir.y, mRayDir.z),
		mSub);//mRayDir.x * mSub.x + 
	c = VectorDot(mSub, mSub) - BASE_SPHERE_RADIUS * BASE_SPHERE_RADIUS;

	float mT = SolveQU(a, b, c);
	if(mT < 1.0f)
		return BACKGROUND;

	float3 mSphereIntrsectPoint = MakePointFromLine(mRayStart, mRayDir, mT);
	float3 mSpherePointNormal = VectorSub(mSphereIntrsectPoint, BASE_SPHERE);
	mSpherePointNormal = NormalizeVector(mSpherePointNormal);

	//mSpherePointNormal.x = mSpherePointNormal.x * 0.5f + 0.5f;
	//mSpherePointNormal.y = mSpherePointNormal.y * 0.5f + 0.5f;
	//mSpherePointNormal.z = mSpherePointNormal.z * 0.5f + 0.5f;

	return make_float4(
		mSpherePointNormal.x,
		mSpherePointNormal.y,
		mSpherePointNormal.z,
		1.0f);
}

__global__ void RayTrace(float4 *mTextureBuffer, int32_t mDimensionX, int32_t mDimensionY)
{
	int gDispatchX = blockIdx.x * blockDim.x + threadIdx.x;
	int mDispatchY = blockIdx.y * blockDim.y + threadIdx.y;

	if (gDispatchX < mDimensionX && mDispatchY < mDimensionY)
	{
		float4 mRay = make_float4((float)gDispatchX / mDimensionX, (float)mDispatchY / mDimensionY, -1.0f, 1.0f);
		mRay.x = 2.0f * mRay.x - 1.0f;
		mRay.y = 2.0f * mRay.y - 1.0f;

		float4 mRayDir = BuildDirection(mRay);
		float4 mRayStart = mWorldPosition[0];
		mRayDir = VectorSub(mRayDir, mRayStart);

		//float ln = sqrt(mRayDir.x * mRayDir.x + mRayDir.y * mRayDir.y + mRayDir.z * mRayDir.z);

		//float mValue = pow(VectorDot(mRayDir, make_float4(0.0f, 0.0f, -1.0f, 0.0f)) / ln, 8.0f);

		mTextureBuffer[mDispatchY * mDimensionX + gDispatchX] = IntersectRaySphere(mRayStart, mRayDir);
			//make_float4(mValue, mValue, mValue, 1.0f);
			//(float)gDispatchX / mDimensionX, (float)mDispatchY / mDimensionY, 1.0f, 1.0f);
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

void TracerCUDA::Tracer::Trace(float *mPosition, float *mMatrix)
{
	float mPos[4] = { mPosition[0], mPosition[1], mPosition[2], 0.0f };

	cudaMemcpyToSymbol(mWorldPosition, mPos, sizeof(float4));
	cudaMemcpyToSymbol(mWorldMatrix, mMatrix, 4 * sizeof(float4));

	dim3 threads(32, 32);
	dim3 blocks((mInfo.mDimX + 31) / 32, (mInfo.mDimY + 31) / 32);
	RayTrace <<<blocks, threads >>> (mCUDA_BUFFER, mInfo.mDimX, mInfo.mDimY);
}