#include "Engine.h"

#pragma once
namespace Window_
{
	class Window
	{
		HGLRC g_hRC;
		HDC g_hDC;
		HWND g_hWnd;
		///////////////////////////////////////////////////////
		SIZE window_size;
		bool sleep_app, size_move;
		///////////////////////////////////////////////////////
		TraceEngine::Engine *mEngine;
		///////////////////////////////////////////////////////
		LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
		///////////////////////////////////////////////////////
		void InitOpenGL(void);
		void InitWindow(HINSTANCE exe_start_adress);
		///////////////////////////////////////////////////////
		void Draw(void);
		///////////////////////////////////////////////////////
	public:
		Window(HINSTANCE exe_start_adress);
		~Window(void);
		///////////////////////////////////////////////////////
		static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
		///////////////////////////////////////////////////////
		int StartApp(void);
		///////////////////////////////////////////////////////
	};
}

