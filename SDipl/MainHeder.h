#pragma once

#if defined(_WIN64) | defined(_WIN32)
//#define GLEW_STATIC
#include <glew.h>
#include <wglew.h>
#include <Windows.h>
#include <stdexcept>

#include <fstream>

#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtc/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>


#endif

