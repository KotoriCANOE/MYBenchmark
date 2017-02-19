#pragma once

// Intrinsics
#include <intrin.h>

// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

// Intrinsic functions
#if defined(__AVX__)
inline __m256 _mm256_abs_ps(const __m256 &x)
{
    static const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x80000000));
    return _mm256_and_ps(x, mask);
}
#endif

#if defined(__SSE2__)
inline __m128 _mm_abs_ps(const __m128 &x)
{
    static const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(~0x80000000));
    return _mm_and_ps(x, mask);
}
#endif
