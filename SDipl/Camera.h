#include "MainHeder.h"
#include <stdint.h>
#pragma once

using namespace glm;
namespace Camera 
{
	#define PI 3.1415926535897932384626433832795f
	#define PIDIV2 PI * 0.5f
	#define TWOPI PI * 2.0f
	class Camera
	{
		/////////////////////////////////
		float near_plane, far_plane;
		/////////////////////////////////
		float aspect;
		unsigned long width, height;
		/////////////////////////////////
		glm::vec3 position;
		/////////////////////////////////
		glm::mat4 view;
		glm::mat4 project;
		/////////////////////////////////
	public:
		Camera(float near_clip_plane, float far_clip_plane, unsigned long width, unsigned long height, float aspect_angle);
		~Camera();

		void UpdateCamera();
		void Resize(int32_t mWidth, int32_t mHeight);
		mat4 GetViewMatrix();
		mat4 GetProjMatrix();
		mat4 GetProjViewMatrix();

		float GetNearPlane();
		float GetFarPlane();

		vec3 GetPosition();
	};
}