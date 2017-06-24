#pragma once

#include <chrono>

// Intrinsics
#if defined(__AVX512__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#elif defined(__SSSE3__)
#include <tmmintrin.h>
#elif defined(__SSE3__)
#include <pmmintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

// chrono
typedef std::chrono::high_resolution_clock MyClock;
typedef std::chrono::duration<double> MySeconds;
typedef std::chrono::duration<double, std::milli> MyMilliseconds;
typedef std::chrono::duration<double, std::micro> MyMicroseconds;
typedef std::chrono::duration<double, std::nano> MyNanoseconds;

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

// Memory allocation

const size_t MEMORY_ALIGNMENT = 64;

inline void *AlignedMalloc(size_t Size, size_t Alignment = MEMORY_ALIGNMENT)
{
    void *Memory = nullptr;
#ifdef _WIN32
    Memory = _aligned_malloc(Size, Alignment);
#else
    if (posix_memalign(&Memory, Alignment, Size))
    {
        Memory = nullptr;
    }
#endif
    return Memory;
}

template < typename _Ty >
void AlignedMalloc(_Ty *&Memory, size_t Count, size_t Alignment = MEMORY_ALIGNMENT)
{
    Memory = reinterpret_cast<_Ty *>(AlignedMalloc(Count * sizeof(_Ty), Alignment));
}


inline void AlignedFree(void **Memory)
{
#ifdef _WIN32
    _aligned_free(*Memory);
#else
    free(*Memory);
#endif
    *Memory = nullptr;
}

template < typename _Ty >
void AlignedFree(_Ty *&Memory)
{
    void *temp = reinterpret_cast<void *>(Memory);
    AlignedFree(&temp);
    Memory = reinterpret_cast<_Ty *>(temp);
}


inline void *AlignedRealloc(void *Memory, size_t NewSize, size_t Alignment = MEMORY_ALIGNMENT)
{
#ifdef _WIN32
    Memory = _aligned_realloc(Memory, NewSize, Alignment);
#else
    AlignedFree(&Memory);
    Memory = AlignedMalloc(NewSize, Alignment);
#endif
    return Memory;
}

template < typename _Ty >
void AlignedRealloc(_Ty *&Memory, size_t NewCount, size_t Alignment = MEMORY_ALIGNMENT)
{
    Memory = reinterpret_cast<_Ty *>(AlignedRealloc(reinterpret_cast<void *>(Memory), NewCount * sizeof(_Ty), Alignment));
}


template < typename _Ty >
size_t CalStride(int width, size_t Alignment = MEMORY_ALIGNMENT)
{
    size_t line_size = static_cast<size_t>(width) * sizeof(_Ty);
    return line_size % Alignment == 0 ? line_size : (line_size / Alignment + 1) * Alignment;
}

