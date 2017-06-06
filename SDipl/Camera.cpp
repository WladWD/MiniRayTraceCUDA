#include "Camera.h"

Camera::Camera::Camera(float near_clip_plane, float far_clip_plane, unsigned long width, unsigned long height, float aspect_angle) : 
	width(width), height(height), position(0.0f, 0.0f, 0.0f),
	aspect((float)width / (float)height), near_plane(near_clip_plane), far_plane(far_clip_plane)
{
	project = glm::perspective(1.0f, aspect, near_clip_plane, far_clip_plane);

	view = glm::mat4(1.0f);
}

Camera::Camera::~Camera()
{
}

void Camera::Camera::Resize(int32_t mWidth, int32_t mHeight)
{
	aspect = (float)mWidth / (float)mHeight;
	project = glm::perspective(1.0f, aspect, near_plane, far_plane);
}

void Camera::Camera::UpdateCamera()
{
	view = glm::mat4(1.0f);
}

glm::mat4 Camera::Camera::GetViewMatrix()
{
	return view;
}

glm::mat4 Camera::Camera::GetProjMatrix()
{
	return project;
}

glm::mat4 Camera::Camera::GetProjViewMatrix()
{
	return project * view;
}

float Camera::Camera::GetNearPlane()
{
	return near_plane;
}

float Camera::Camera::GetFarPlane()
{
	return far_plane;
}

glm::vec3 Camera::Camera::GetPosition()
{
	return position;
}