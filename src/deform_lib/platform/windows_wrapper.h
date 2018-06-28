#pragma once

#ifdef _WIN32
#if defined(_WINDOWS_)
#error "<windows.h> has been included by other means than windows_wrapper.h"
#endif

#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
#define NODRAWTEXT // DrawText()

#include <windows.h>

// Undefine annoying windows macros
#undef min
#undef max
#undef MB_RIGHT

#endif // _WIN32
