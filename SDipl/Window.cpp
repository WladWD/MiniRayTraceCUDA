#include "Window.h"


Window_::Window::Window(HINSTANCE exe_start_adress)
{
	mEngine = 0;
	sleep_app = true;
	sleep_app = size_move = false;
	InitWindow(exe_start_adress);
	InitOpenGL();

	mEngine = new TraceEngine::Engine(128, 128);

	mEngine->Resize(window_size.cx, window_size.cy);
}

Window_::Window::~Window()
{
	if (g_hRC)
	{
		wglMakeCurrent(g_hDC, 0);
		wglDeleteContext(g_hRC);
		g_hRC = 0;
	}

	if (g_hDC)
	{
		ReleaseDC(g_hWnd, g_hDC);
		g_hDC = 0;
	}

	delete mEngine;
}

void Window_::Window::InitOpenGL(void)
{
	int pixelformat_find;
	PIXELFORMATDESCRIPTOR pixel_format = { 0 };

	g_hDC = GetDC(g_hWnd);

	pixel_format.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pixel_format.nVersion = 1;
	pixel_format.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pixel_format.iPixelType = PFD_TYPE_RGBA;
	pixel_format.cColorBits = 32;
	pixel_format.cDepthBits = 32;//24
	pixel_format.cStencilBits = 0;

	if (!(pixelformat_find = ChoosePixelFormat(g_hDC, &pixel_format)))
	{
		MessageBox(NULL, L"Eror Choose Pixel Format!!!", L"Error", MB_OK || MB_ICONERROR);
		//error
		throw L"Eror Choose Pixel Format";
	}

	if (!SetPixelFormat(g_hDC, pixelformat_find, &pixel_format))
	{
		MessageBox(NULL, L"Eror Set Pixel Format!!!", L"Error", MB_OK || MB_ICONERROR);
		//error
		throw L"Eror Set Pixel Format";
	}
	
	g_hRC = wglCreateContext(g_hDC);
	wglMakeCurrent(g_hDC, g_hRC);

	GLenum glew_init_res = glewInit();
	if (glew_init_res != GLEW_OK)
	{
		MessageBox(NULL, L"Eror GLEW init!!!", L"Error", MB_OK || MB_ICONERROR);
		//error
		throw L"Eror GLEW init";
	}

	if (!GLEW_VERSION_4_0)//GLEW_VERSION_4_4)
	{
		MessageBox(NULL, L"Eror OpenGL 4.0 is not support", L"Error", MB_OK || MB_ICONERROR);
		//error
		throw L"OpenGL 4.0 is not support";
	}

	glEnable(GL_TEXTURE_2D);
}

void Window_::Window::InitWindow(HINSTANCE exe_start_adress)
{
	WNDCLASSEX wndclass;
	ZeroMemory(&wndclass, sizeof(wndclass));

	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.cbSize = sizeof(wndclass);
	wndclass.hbrBackground = CreateSolidBrush(RGB(0x1f, 0xbf, 0xef));
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIcon = LoadIcon(NULL, IDI_WINLOGO);
	wndclass.hIconSm = NULL;
	wndclass.hInstance = exe_start_adress;
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = L"Tracer";
	wndclass.lpszMenuName = NULL;
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;

	HDC hdc_cur = GetDC(NULL);

	UINT width = GetDeviceCaps(hdc_cur, VERTRES);
	UINT height = GetDeviceCaps(hdc_cur, HORZRES);
	DeleteDC(hdc_cur);

	if (!RegisterClassEx(&wndclass))
	{
		MessageBox(NULL, L"Error Register Window", NULL, MB_OK);
		exit(1);
	}
	// WS_OVERLAPPEDWINDOW WS_POPUP
	if ((g_hWnd = CreateWindowEx(NULL, wndclass.lpszClassName, 0, WS_OVERLAPPEDWINDOW, 0, 0, width, height, NULL, NULL, exe_start_adress, this)) == NULL)
	{
		MessageBox(NULL, L"Error Create Window", NULL, MB_OK);
		exit(1);
	}

	ShowWindow(g_hWnd, SW_SHOWMAXIMIZED);
	ShowCursor(FALSE);
}

LRESULT Window_::Window::WindowProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
	switch (msg)
	{
	case WM_DESTROY:
	{
		PostQuitMessage(0);
		return 0;
	}break;
	case WM_KEYDOWN: 
	{
		if (wp == VK_ESCAPE)DestroyWindow(hwnd);
		if (wp == VK_ESCAPE)DestroyWindow(hwnd);

		return 0;
	}break;
	case WM_PAINT:
	{
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hwnd, &ps);

		EndPaint(hwnd, &ps);
		return 0;
	}break;
	case WM_CLOSE:
	{
		DestroyWindow(hwnd);
	}return 0;
	case WM_SIZE:
	{
		window_size.cx = LOWORD(lp);
		window_size.cy = HIWORD(lp);

		if(mEngine)
			mEngine->Resize(window_size.cx, window_size.cy);
	}return 0;
	case WM_ENTERSIZEMOVE:
	{
		sleep_app = true;
		size_move = true;
	}return 0;
	case WM_EXITSIZEMOVE:
	{
		sleep_app = false;
		size_move = false;
	}return 0;
	case WM_KEYUP: 
	{
	}return 0;
	case WM_ACTIVATE:
	{

		if (LOWORD(wp) == WA_INACTIVE)
		{
			sleep_app = true;

		}
		else
		{
			sleep_app = false;
		}
	}return 0;
	default: return DefWindowProc(hwnd, msg, wp, lp);
	}
}

LRESULT Window_::Window::WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
	static Window_::Window *lp_fn_class_wnd_proc = NULL;
	if (lp_fn_class_wnd_proc)
	{
		return lp_fn_class_wnd_proc->WindowProc(hwnd, msg, wp, lp);
	}
	if (msg == WM_CREATE)
	{
		lp_fn_class_wnd_proc = (Window_::Window *)(((CREATESTRUCT *)lp)->lpCreateParams);
		return lp_fn_class_wnd_proc->WindowProc(hwnd, msg, wp, lp);
	}
	else return DefWindowProc(hwnd, msg, wp, lp);
}

int Window_::Window::StartApp(void)
{
	MSG msg = { 0 };
	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			if (sleep_app)Sleep(120);
			else
			{
				Draw();
				//ComputeFPS();
			}
		}
	}
	return (int)msg.wParam;
}


void Window_::Window::Draw()
{
	mEngine->Tick();
	mEngine->Draw();

	glFlush();
	SwapBuffers(g_hDC);
}