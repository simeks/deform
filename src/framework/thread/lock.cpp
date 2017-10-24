#include "lock.h"

namespace thread
{
#ifdef DF_PLATFORM_WINDOWS
    CriticalSection::CriticalSection()
    {
        InitializeCriticalSection(&_cs);
    }
    CriticalSection::~CriticalSection()
    {
        DeleteCriticalSection(&_cs);
    }

    void CriticalSection::lock()
    {
        EnterCriticalSection(&_cs);
    }

    void CriticalSection::unlock()
    {
        LeaveCriticalSection(&_cs);
    }

    bool CriticalSection::try_lock()
    {
        return TryEnterCriticalSection(&_cs) != FALSE;
    }
#endif
}
