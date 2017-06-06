#include "Window.h"

int WINAPI WinMain(HINSTANCE exe_start_adress, HINSTANCE old_value, LPSTR cmd, int show)
{
	UNREFERENCED_PARAMETER(old_value);
	UNREFERENCED_PARAMETER(cmd);

	Window_::Window *window = new Window_::Window(exe_start_adress);

	return window->StartApp();
}