// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

template<class T, typename U = unsigned> __host__ __device__ __forceinline__ T& pow_byref(T& val, U p)
{
    T sqr = val;
    val = T::csel(val, T::one(), p&1);

    #pragma unroll 1
    while (p >>= 1) {
        sqr.sqr();
        if (p & 1) val *= sqr;
    }

    return val;
}

// This is meant to be used for code size optimization by deduplicating otherwise inlined pow_byref.
template<class T> __device__ __noinline__ T pow_byval(T val, unsigned p) { return pow_byref(val, p); }

#include <cassert>

// Raise to a constant power, e.g. x^7. The idea is to let compiler "decide" how to unroll with expectation that for small constants it will be fully inrolled.
template<class T> __host__ __device__ __forceinline__ T& pow_byref(T& val, int p)
{
    assert(p >= 2);

    T sqr = val;
    if ((p & 1) == 0) {
        do {
            sqr.sqr();
            p >>= 1;
        } while ((p & 1) == 0);
        val = sqr;
    }
    for (p >>= 1; p; p >>= 1) {
        sqr.sqr();
        if (p & 1) val *= sqr;
    }
    return val;
}

// This is meant to be used for code size optimization by deduplicating otherwise inlined pow_byref.

template<class T> __device__ __noinline__ T pow_byval(T val, int p) { return pow_byref(val, p); }