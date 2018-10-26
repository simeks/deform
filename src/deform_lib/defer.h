#pragma once

#include <functional>

// defer macro for defering function calls as in golang
// Automatically invokes code when leaving a scope.
// Note: In comparison to golang the defer will be invoked when leaving any
//       scope, not only when leaving a function.
// Example:
// void func()
// {
//     init();
//     defer{shutdown();};
//     do stuff...
// }
// Here shutdown() will be called when leaving func()

struct DeferCall {
    std::function<void()> _f;

    template<typename F>
    DeferCall(F &&f) : _f(std::forward<F>(f)) {}
    DeferCall(DeferCall&& o) : _f(std::move(o._f))
    {
        _f = nullptr;
    }
    ~DeferCall()
    {
        if (_f)
            _f();
    }

    DeferCall(const DeferCall&) = delete;
    void operator=(const DeferCall&) = delete;
};

#define DEFER_NAME(a, b) a##b
#define DEFER_NAME2(a, b) DEFER_NAME(a, b)
#define defer DeferCall DEFER_NAME2(defer__, __LINE__) = [&]()
