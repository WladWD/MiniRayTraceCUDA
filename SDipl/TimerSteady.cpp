#include "TimerSteady.h"


Timer::TimerSteady::TimerSteady() : mGlobalTime(0.0f), mDelataTime(0.0f), mSleep(false)
{
	mLast = mCurrent = std::chrono::steady_clock::now();
}

Timer::TimerSteady::~TimerSteady()
{
}

float Timer::TimerSteady::GetGlobalTime(void)
{
	return mGlobalTime;
}

float Timer::TimerSteady::GetDelataTime(void)
{
	return mDelataTime;
}

void Timer::TimerSteady::Tick(void)
{
	static const float mDiv = 1.0f / 1000.0f;
	if (!mSleep)
	{
		mLast = mCurrent;
		mCurrent = std::chrono::steady_clock::now();

		mDelataTime =
			static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(mCurrent - mLast).count()) * mDiv;

		mDelataTime = mDelataTime > 0.4f ? mDelataTime : 0.4f;

		mGlobalTime += mDelataTime;
	}
	else mDelataTime = 0.0f;
}

void Timer::TimerSteady::Resume(void)
{
	if (mSleep)
	{
		mSleep = false;
		mLast = mCurrent = std::chrono::steady_clock::now();
	}
}

void Timer::TimerSteady::Pause(void)
{
	if (!mSleep)
		mSleep = true;
}
