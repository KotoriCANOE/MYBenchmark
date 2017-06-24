#pragma once

#include "utils.h"
#include <string>
#include <algorithm>
#include <iostream>
#include <iomanip>

class InstructionTest
{
public:
    bool silent = false;
    int batch = 0x400000;
    size_t length = 0x1000000;
    int threads = 0;
    int loop = 0x1000;
    int type = 1;

protected:
    bool stress_test;
    int times;
    size_t _length;
    float *vecA = nullptr;
    float *vecB = nullptr;
    float *vecC = nullptr;
    float *vecD = nullptr;

public:
    virtual ~InstructionTest() {}

    void RunTest()
    {
        // Standard I/O
        const std::streamsize io_precision_origin = std::cout.precision();
        std::fixed(std::cout);

        // OpenMP
#ifdef _OPENMP
        const int threads_origin = omp_get_max_threads();
        const int threads_new = threads > 0 ? threads : std::max(1, omp_get_num_procs() - threads);
        omp_set_num_threads(threads_new);
#else
        const int threads_new = 1;
#endif

        // Stress Test
        stress_test = false;

        if (loop == 0)
        {
            loop = threads_new;
            stress_test = true;

            if (!silent) std::cout << "\nRunning stress test...";
        }

        // Kernel
        times = 0;

        // initialize
        switch (type)
        {
        case 1:
            _length = length * 16;
            break;
        case 2:
            _length = length * 3;
            AlignedMalloc(vecA, _length, simdWidth());
            break;
        case 3:
            _length = length;
            AlignedMalloc(vecA, _length, simdWidth());
            AlignedMalloc(vecB, _length, simdWidth());
            break;
        default:
            _length = length;
            break;
        }

        // run the tests
        while (true)
        { // infinite loop for continuous tests
            // start time
            MyClock::time_point t1 = MyClock::now();

            // start kernel
            kernel();

            // end time
            MySeconds time_span = std::chrono::duration_cast<MySeconds>(MyClock::now() - t1);
            ++times;

            // output
            if (!silent)
            {
                output(time_span);
            }
        }

        // free
        switch (type)
        {
        case 2:
            AlignedFree(vecA);
            break;
        case 3:
            AlignedFree(vecA);
            AlignedFree(vecB);
            break;
        default:
            break;
        }

        // OpenMP
#ifdef _OPENMP
        omp_set_num_threads(threads_origin);
#endif

        // reset I/O parameters
        std::cout << std::setprecision(io_precision_origin);
        std::defaultfloat(std::cout);
    }

protected:
    virtual size_t simdWidth() const = 0;

    virtual void kernel() const = 0;

    virtual void output(const MySeconds &time_span) const
    {
        std::cout << std::setprecision(6)
            << times << ": It took " << time_span.count()
            << " seconds to run " << loop << " loops.\n";

        switch (type)
        {
        case 1:
        case 2:
        case 3:
            std::cout << std::setprecision(6)
                << "    Achieving "
                << 2 * _length * loop / std::chrono::duration_cast<MyNanoseconds>(time_span).count()
                << " GFLOPS (single precision).\n";
            break;
        case 4:
        case 5:
            std::cout << std::setprecision(3)
                << "    Average batch time (per loop) is "
                << std::chrono::duration_cast<MyMicroseconds>(time_span).count() / loop
                << " microseconds.\n";
            break;
        default:
            break;
        }
    }
};



class AVXTest
    : public InstructionTest
{
public:
    static const size_t simd_width = 32;

protected:
    virtual size_t simdWidth() const override { return simd_width; }

    virtual void kernel() const override
    {
#pragma omp parallel for
        for (int l = 0; l < loop; ++l)
        { // main loop
            do switch (type)
            {
            case 1:
            {
                static const int batch = 8;
                static const size_t simd_step = simd_width * batch / sizeof(float);
                __m256 r0 = _mm256_setzero_ps();
                __m256 r1 = _mm256_setzero_ps();
                __m256 r2 = _mm256_setzero_ps();
                __m256 r3 = _mm256_setzero_ps();
                __m256 r4 = _mm256_setzero_ps();
                __m256 r5 = _mm256_setzero_ps();
                __m256 r6 = _mm256_setzero_ps();
                __m256 r7 = _mm256_setzero_ps();

                for (size_t i = 0; i < _length; i += simd_step)
                {
                    r0 = _mm256_mul_ps(r0, r0); r0 = _mm256_add_ps(r0, r0);
                    r1 = _mm256_mul_ps(r1, r1); r1 = _mm256_add_ps(r1, r1);
                    r2 = _mm256_mul_ps(r2, r2); r2 = _mm256_add_ps(r2, r2);
                    r3 = _mm256_mul_ps(r3, r3); r3 = _mm256_add_ps(r3, r3);
                    r4 = _mm256_mul_ps(r4, r4); r4 = _mm256_add_ps(r4, r4);
                    r5 = _mm256_mul_ps(r5, r5); r5 = _mm256_add_ps(r5, r5);
                    r6 = _mm256_mul_ps(r6, r6); r6 = _mm256_add_ps(r6, r6);
                    r7 = _mm256_mul_ps(r7, r7); r7 = _mm256_add_ps(r7, r7);
                }

                alignas(simd_width) float mem[simd_width * batch];
                _mm256_store_ps(mem + simd_width * 0x0, r0);
                _mm256_store_ps(mem + simd_width * 0x1, r1);
                _mm256_store_ps(mem + simd_width * 0x2, r2);
                _mm256_store_ps(mem + simd_width * 0x3, r3);
                _mm256_store_ps(mem + simd_width * 0x4, r4);
                _mm256_store_ps(mem + simd_width * 0x5, r5);
                _mm256_store_ps(mem + simd_width * 0x6, r6);
                _mm256_store_ps(mem + simd_width * 0x7, r7);

                break;
            }
            case 2:
            {
                static const int batch = 4;
                static const size_t simd_step1 = simd_width / sizeof(float);
                static const size_t simd_step2 = simd_step1 * batch;

                const float *vecA0 = vecA + simd_step1 * 0;
                const float *vecA1 = vecA + simd_step1 * 1;
                const float *vecA2 = vecA + simd_step1 * 2;
                const float *vecA3 = vecA + simd_step1 * 3;

                __m256 b0 = _mm256_setzero_ps();
                __m256 b1 = _mm256_setzero_ps();
                __m256 b2 = _mm256_setzero_ps();
                __m256 b3 = _mm256_setzero_ps();

                for (size_t i = 0; i < _length; i += simd_step2)
                {
                    const __m256 a0 = _mm256_load_ps(vecA0 + i);
                    const __m256 a1 = _mm256_load_ps(vecA1 + i);
                    const __m256 a2 = _mm256_load_ps(vecA2 + i);
                    const __m256 a3 = _mm256_load_ps(vecA3 + i);

                    b0 = _mm256_add_ps(_mm256_mul_ps(a0, a0), b0);
                    b1 = _mm256_add_ps(_mm256_mul_ps(a1, a1), b1);
                    b2 = _mm256_add_ps(_mm256_mul_ps(a2, a2), b2);
                    b3 = _mm256_add_ps(_mm256_mul_ps(a3, a3), b3);
                }

                const __m256 b = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
                alignas(simd_width) float mem[simd_width];
                _mm256_store_ps(mem, b);

                break;
            }
            case 3:
            {
                static const size_t simd_step = simd_width / sizeof(float);

                for (size_t i = 0; i < _length; i += simd_step)
                {
                    const __m256 a = _mm256_load_ps(vecA + i);
                    const __m256 b = _mm256_add_ps(_mm256_mul_ps(a, a), a);
                    _mm256_store_ps(vecB + i, b);
                }

                break;
            }
            case 4:
            {
                const __m256 c0 = _mm256_setzero_ps();
                const __m256 c1 = _mm256_set1_ps(1);
                const __m256 c2 = _mm256_set_ps(1, 2, 4, 8, 16, 32, 64, 128);
                const __m256 c3 = _mm256_set_ps(128, 64, 32, 16, 8, 4, 2, 1);

                __m256 r0 = c2;
                __m256 r1 = c3;
                __m256 r2, r3;

                for (int j = 0; j < batch; ++j)
                { // loop for batch
                  // Arithmetic
                    r2 = _mm256_add_ps(r0, r1);
                    r3 = _mm256_sub_ps(r0, r1);
                    r0 = _mm256_mul_ps(r2, r3);
                    // Special Math Functions
                    r2 = _mm256_min_ps(r0, r1);
                    r3 = _mm256_max_ps(r0, r1);
                    // Swizzle
                    r0 = _mm256_unpacklo_ps(r2, r3);
                    r1 = _mm256_unpackhi_ps(r2, r3);
                }

                alignas(simd_width) float mem[simd_width / 2];
                _mm256_store_ps(mem, r0);
                _mm256_store_ps(mem + simd_width / 4, r1);

                break;
            }
            case 5:
            {
                const __m256 c0 = _mm256_setzero_ps();
                const __m256 c1 = _mm256_set1_ps(1);
                const __m256 c2 = _mm256_set_ps(1, 2, 4, 8, 16, 32, 64, 128);
                const __m256 c3 = _mm256_set_ps(128, 64, 32, 16, 8, 4, 2, 1);

                __m256 r0 = c2;
                __m256 r1 = c3;
                __m256 r2, r3;

                for (int j = 0; j < batch; ++j)
                { // loop for batch
                  // Arithmetic
                    r2 = _mm256_add_ps(r0, r1);
                    r3 = _mm256_sub_ps(r0, r1);
                    r0 = _mm256_hadd_ps(r2, r3);
                    r1 = _mm256_mul_ps(r2, r3);
                    // Logical
                    r2 = _mm256_and_ps(r0, r1);
                    r3 = _mm256_or_ps(r0, r1);
                    r0 = _mm256_andnot_ps(r2, r3);
                    r1 = _mm256_xor_ps(r2, r3);
                    // Special Math Functions
                    r2 = _mm256_min_ps(r0, r1);
                    r3 = _mm256_max_ps(r0, r1);
                    r0 = _mm256_floor_ps(r2);
                    r1 = _mm256_ceil_ps(r3);
                    // Swizzle
                    r2 = _mm256_unpackhi_ps(r0, r1);
                    r3 = _mm256_unpacklo_ps(r0, r1);
                    r0 = _mm256_shuffle_ps(r2, r3, 0xaa);
                    r1 = _mm256_blend_ps(r2, r3, 0x55);
                }

                alignas(simd_width) float mem[simd_width / 2];
                _mm256_store_ps(mem, r0);
                _mm256_store_ps(mem + simd_width / 4, r1);

                break;
            }
            default:
                if (!silent) std::cout << "type=" << type << " is not supported by this mode!";
                break;
            } while (stress_test); // infinite loop when doing stress test
        }
    }
};


class AVX2Test
    : public InstructionTest
{
public:
    static const size_t simd_width = 32;

protected:
    virtual size_t simdWidth() const override { return simd_width; }

    virtual void kernel() const override
    {
#pragma omp parallel for
        for (int l = 0; l < loop; ++l)
        { // main loop
            do switch (type)
            {
            case 1:
            {
                static const int batch = 8;
                static const size_t simd_step = simd_width * batch / sizeof(float);
                __m256 r0 = _mm256_setzero_ps();
                __m256 r1 = _mm256_setzero_ps();
                __m256 r2 = _mm256_setzero_ps();
                __m256 r3 = _mm256_setzero_ps();
                __m256 r4 = _mm256_setzero_ps();
                __m256 r5 = _mm256_setzero_ps();
                __m256 r6 = _mm256_setzero_ps();
                __m256 r7 = _mm256_setzero_ps();

                for (size_t i = 0; i < _length; i += simd_step)
                {
                    r0 = _mm256_fmadd_ps(r0, r0, r0);
                    r1 = _mm256_fmadd_ps(r1, r1, r1);
                    r2 = _mm256_fmadd_ps(r2, r2, r2);
                    r3 = _mm256_fmadd_ps(r3, r3, r3);
                    r4 = _mm256_fmadd_ps(r4, r4, r4);
                    r5 = _mm256_fmadd_ps(r5, r5, r5);
                    r6 = _mm256_fmadd_ps(r6, r6, r6);
                    r7 = _mm256_fmadd_ps(r7, r7, r7);
                }

                alignas(simd_width) float mem[simd_width * batch];
                _mm256_store_ps(mem + simd_width * 0x0, r0);
                _mm256_store_ps(mem + simd_width * 0x1, r1);
                _mm256_store_ps(mem + simd_width * 0x2, r2);
                _mm256_store_ps(mem + simd_width * 0x3, r3);
                _mm256_store_ps(mem + simd_width * 0x4, r4);
                _mm256_store_ps(mem + simd_width * 0x5, r5);
                _mm256_store_ps(mem + simd_width * 0x6, r6);
                _mm256_store_ps(mem + simd_width * 0x7, r7);

                break;
            }
            case 2:
            {
                static const int batch = 4;
                static const size_t simd_step1 = simd_width / sizeof(float);
                static const size_t simd_step2 = simd_step1 * batch;

                const float *vecA0 = vecA + simd_step1 * 0;
                const float *vecA1 = vecA + simd_step1 * 1;
                const float *vecA2 = vecA + simd_step1 * 2;
                const float *vecA3 = vecA + simd_step1 * 3;

                __m256 b0 = _mm256_setzero_ps();
                __m256 b1 = _mm256_setzero_ps();
                __m256 b2 = _mm256_setzero_ps();
                __m256 b3 = _mm256_setzero_ps();

                for (size_t i = 0; i < _length; i += simd_step2)
                {
                    const __m256 a0 = _mm256_load_ps(vecA0 + i);
                    const __m256 a1 = _mm256_load_ps(vecA1 + i);
                    const __m256 a2 = _mm256_load_ps(vecA2 + i);
                    const __m256 a3 = _mm256_load_ps(vecA3 + i);

                    b0 = _mm256_fmadd_ps(a0, a0, b0);
                    b1 = _mm256_fmadd_ps(a1, a1, b1);
                    b2 = _mm256_fmadd_ps(a2, a2, b2);
                    b3 = _mm256_fmadd_ps(a3, a3, b3);
                }

                const __m256 b = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
                alignas(simd_width) float mem[simd_width];
                _mm256_store_ps(mem, b);

                break;
            }
            case 3:
            {
                static const size_t simd_step = simd_width / sizeof(float);

                for (size_t i = 0; i < _length; i += simd_step)
                {
                    const __m256 a = _mm256_load_ps(vecA + i);
                    const __m256 b = _mm256_fmadd_ps(a, a, a);
                    _mm256_store_ps(vecB + i, b);
                }

                break;
            }
            case 4:
            {
                const __m256i c0 = _mm256_setzero_si256();
                const __m256i c1 = _mm256_set1_epi32(1);
                const __m256i c2 = _mm256_set_epi32(1, 2, 4, 8, 16, 32, 64, 128);
                const __m256i c3 = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);

                __m256i r0 = c2;
                __m256i r1 = c3;
                __m256i r2, r3;

                for (int j = 0; j < batch; ++j)
                { // loop for batch
                  // Arithmetic
                    r2 = _mm256_add_epi32(r0, r1);
                    r3 = _mm256_sub_epi32(r0, r1);
                    r0 = _mm256_mul_epi32(r2, r3);
                    // Special Math Functions
                    r2 = _mm256_min_epi32(r0, r1);
                    r3 = _mm256_max_epi32(r0, r1);
                    // Swizzle
                    r0 = _mm256_unpacklo_epi32(r2, r3);
                    r1 = _mm256_unpackhi_epi32(r2, r3);
                }

                alignas(simd_width) int32_t mem[simd_width / 2];
                _mm256_store_si256(reinterpret_cast<__m256i *>(mem), r0);
                _mm256_store_si256(reinterpret_cast<__m256i *>(mem + simd_width / 4), r1);

                break;
            }
            case 5:
            {
                const __m256i c0 = _mm256_setzero_si256();
                const __m256i c1 = _mm256_set1_epi32(1);
                const __m256i c2 = _mm256_set_epi32(1, 2, 4, 8, 16, 32, 64, 128);
                const __m256i c3 = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);

                __m256i r0 = c2;
                __m256i r1 = c3;
                __m256i r2, r3;

                for (int j = 0; j < batch; ++j)
                { // loop for batch
                  // Arithmetic
                    r2 = _mm256_add_epi32(r0, r1);
                    r3 = _mm256_sub_epi32(r0, r1);
                    r0 = _mm256_hadd_epi32(r2, r3);
                    r1 = _mm256_mul_epi32(r2, r3);
                    // Logical
                    r2 = _mm256_and_si256(r0, r1);
                    r3 = _mm256_or_si256(r0, r1);
                    r0 = _mm256_andnot_si256(r2, r3);
                    r1 = _mm256_xor_si256(r2, r3);
                    // Special Math Functions
                    r2 = _mm256_min_epi32(r0, r1);
                    r3 = _mm256_max_epi32(r0, r1);
                    // Swizzle
                    r2 = _mm256_unpackhi_epi32(r0, r1);
                    r3 = _mm256_unpacklo_epi32(r0, r1);
                    r0 = _mm256_unpacklo_epi32(r2, r3);
                    r1 = _mm256_blend_epi32(r2, r3, 0x55);
                }

                alignas(simd_width) int32_t mem[simd_width / 2];
                _mm256_store_si256(reinterpret_cast<__m256i *>(mem), r0);
                _mm256_store_si256(reinterpret_cast<__m256i *>(mem + simd_width / 4), r1);

                break;
            }
            default:
                if (!silent) std::cout << "type=" << type << " is not supported by this mode!";
                break;
            } while (stress_test); // infinite loop when doing stress test
        }
    }
};

class AVX512FTest
    : public InstructionTest
{
public:
    static const size_t simd_width = 64;

protected:
    virtual size_t simdWidth() const override { return simd_width; }

    virtual void kernel() const override
    {
#pragma omp parallel for
        for (int l = 0; l < loop; ++l)
        { // main loop
            do switch (type)
            {
            case 1:
            {
                static const int batch = 8;
                static const size_t simd_step = simd_width * batch / sizeof(float);
                __m512 r0 = _mm512_setzero_ps();
                __m512 r1 = _mm512_setzero_ps();
                __m512 r2 = _mm512_setzero_ps();
                __m512 r3 = _mm512_setzero_ps();
                __m512 r4 = _mm512_setzero_ps();
                __m512 r5 = _mm512_setzero_ps();
                __m512 r6 = _mm512_setzero_ps();
                __m512 r7 = _mm512_setzero_ps();

                for (size_t i = 0; i < _length; i += simd_step)
                {
                    r0 = _mm512_fmadd_ps(r0, r0, r0);
                    r1 = _mm512_fmadd_ps(r1, r1, r1);
                    r2 = _mm512_fmadd_ps(r2, r2, r2);
                    r3 = _mm512_fmadd_ps(r3, r3, r3);
                    r4 = _mm512_fmadd_ps(r4, r4, r4);
                    r5 = _mm512_fmadd_ps(r5, r5, r5);
                    r6 = _mm512_fmadd_ps(r6, r6, r6);
                    r7 = _mm512_fmadd_ps(r7, r7, r7);
                }

                alignas(simd_width) float mem[simd_width * batch];
                _mm512_store_ps(mem + simd_width * 0x0, r0);
                _mm512_store_ps(mem + simd_width * 0x1, r1);
                _mm512_store_ps(mem + simd_width * 0x2, r2);
                _mm512_store_ps(mem + simd_width * 0x3, r3);
                _mm512_store_ps(mem + simd_width * 0x4, r4);
                _mm512_store_ps(mem + simd_width * 0x5, r5);
                _mm512_store_ps(mem + simd_width * 0x6, r6);
                _mm512_store_ps(mem + simd_width * 0x7, r7);

                break;
            }
            case 2:
            {
                static const int batch = 4;
                static const size_t simd_step1 = simd_width / sizeof(float);
                static const size_t simd_step2 = simd_step1 * batch;

                const float *vecA0 = vecA + simd_step1 * 0;
                const float *vecA1 = vecA + simd_step1 * 1;
                const float *vecA2 = vecA + simd_step1 * 2;
                const float *vecA3 = vecA + simd_step1 * 3;

                __m512 b0 = _mm512_setzero_ps();
                __m512 b1 = _mm512_setzero_ps();
                __m512 b2 = _mm512_setzero_ps();
                __m512 b3 = _mm512_setzero_ps();

                for (size_t i = 0; i < _length; i += simd_step2)
                {
                    const __m512 a0 = _mm512_load_ps(vecA0 + i);
                    const __m512 a1 = _mm512_load_ps(vecA1 + i);
                    const __m512 a2 = _mm512_load_ps(vecA2 + i);
                    const __m512 a3 = _mm512_load_ps(vecA3 + i);

                    b0 = _mm512_fmadd_ps(a0, a0, b0);
                    b1 = _mm512_fmadd_ps(a1, a1, b1);
                    b2 = _mm512_fmadd_ps(a2, a2, b2);
                    b3 = _mm512_fmadd_ps(a3, a3, b3);
                }

                const __m512 b = _mm512_add_ps(_mm512_add_ps(b0, b1), _mm512_add_ps(b2, b3));
                alignas(simd_width) float mem[simd_width];
                _mm512_store_ps(mem, b);

                break;
            }
            case 3:
            {
                static const size_t simd_step = simd_width / sizeof(float);

                for (size_t i = 0; i < _length; i += simd_step)
                {
                    const __m512 a = _mm512_load_ps(vecA + i);
                    const __m512 b = _mm512_fmadd_ps(a, a, a);
                    _mm512_store_ps(vecB + i, b);
                }

                break;
            }
            case 4:
            {
                const __m512 c0 = _mm512_setzero_ps();
                const __m512 c1 = _mm512_set1_ps(1);
                const __m512 c2 = _mm512_set_ps(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16184, 32768);
                const __m512 c3 = _mm512_set_ps(32768, 16184, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1);

                __m512 r0 = c2;
                __m512 r1 = c3;
                __m512 r2, r3;

                for (int j = 0; j < batch; ++j)
                { // loop for batch
                  // Arithmetic
                    r2 = _mm512_add_ps(r0, r1);
                    r3 = _mm512_sub_ps(r0, r1);
                    r0 = _mm512_mul_ps(r2, r3);
                    // Special Math Functions
                    r2 = _mm512_min_ps(r0, r1);
                    r3 = _mm512_max_ps(r0, r1);
                    // Swizzle
                    r0 = _mm512_shuffle_ps(r2, r3, 0xaa);
                }

                alignas(simd_width) float mem[simd_width / 2];
                _mm512_store_ps(mem, r0);
                _mm512_store_ps(mem + simd_width / 4, r1);

                break;
            }
            default:
                if (!silent) std::cout << "type=" << type << " is not supported by this mode!";
                break;
            } while (stress_test); // infinite loop when doing stress test
        }
    }
};
