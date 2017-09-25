// Copyright 2008-2014 Simon Ekström

#include "../thread.h"

#include <pthread.h>


long thread::interlocked_increment(long volatile* addend)
{
	return __sync_fetch_and_add(addend, 1) + 1;
}
long thread::interlocked_decrement(long volatile* addend)
{
	return __sync_fetch_and_sub(addend, 1) - 1;
}
long thread::interlocked_exchange_add(long volatile* addend, long value)
{
	return __sync_fetch_and_add(addend, value);
}
unsigned long thread::interlocked_exchange_subtract(unsigned long volatile* addend, unsigned long value)
{
	return __sync_fetch_and_sub(addend, value);
}
long thread::interlocked_compare_exchange(long volatile* dest, long exchange, long comparand)
{
	return __sync_val_compare_and_swap(dest, comparand, exchange);
}
long thread::interlocked_exchange(long volatile* dest, long value)
{
	return __sync_val_compare_and_swap(dest, *dest, value);
}

int64_t thread::interlocked_increment_64(int64_t volatile* addend)
{
	return __sync_fetch_and_add(addend, 1) + 1;
}
int64_t thread::interlocked_decrement_64(int64_t volatile* addend)
{
	return __sync_fetch_and_sub(addend, 1) - 1;
}

int64_t thread::interlocked_exchange_add_64(int64_t volatile* addend, int64_t value)
{
	return __sync_fetch_and_add(addend, value);
}



