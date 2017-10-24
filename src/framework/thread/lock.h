#pragma once

#ifdef DF_PLATFORM_WINDOWS
    #include "platform/windows_wrapper.h"
#endif

namespace thread
{
    /// @brief Critical section lock for win32
    class CriticalSection
    {
    public:
        CriticalSection();
        ~CriticalSection();

        void lock();
        void unlock();
        bool try_lock();

    private:
        #ifdef DF_PLATFORM_WINDOWS
            CRITICAL_SECTION _cs;
        #endif
    };

    template<typename TLock>
    class ScopedLock
    {
    public:
        ScopedLock(TLock& lock) : _lock(lock)
        {
            _lock.lock();
        }
        ~ScopedLock()
        {
            _lock.unlock();
        }

    private:
        TLock& _lock;

        ScopedLock();
        ScopedLock(const ScopedLock&);
        ScopedLock& operator=(const ScopedLock&);
    };
}
