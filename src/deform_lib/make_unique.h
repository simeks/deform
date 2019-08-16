#pragma once

#include <memory>

#if (defined(_MSC_VER) && _MSC_VER < 1800) \
    || (defined(__GNUC__) && __cplusplus < 201402L)

    template<typename T, typename ...Args>
    std::unique_ptr<T> make_unique( Args&& ...args )
    {
        return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
    }

#else
    using std::make_unique;
#endif
