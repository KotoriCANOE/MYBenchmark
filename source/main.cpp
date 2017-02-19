#include "utils.h"
#include <string>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>

typedef std::chrono::high_resolution_clock MyClock;
typedef std::chrono::duration<double> MySeconds;
typedef std::chrono::duration<double, std::milli> MyMilliseconds;
typedef std::chrono::duration<double, std::micro> MyMicroseconds;

// AVX Operator Test
void AVXOperatorTest(int threads, int loop)
{
    // Standard I/O
    const std::streamsize io_precision_origin = std::cout.precision();
    std::cout.setf(std::ios::fixed, std::ios::floatfield); // floatfield set to fixed

    // Constants
    static const int batch = 0x20000;

#ifdef _OPENMP
    const int threads_origin = omp_get_max_threads();
    const int threads_new = threads > 0 ? threads : std::max(1, omp_get_num_procs() - threads);
    omp_set_num_threads(threads_new);
#else
    const int threads_new = 1;
#endif

    // Stress Test
    bool stress_test = false;

    if (loop == 0)
    {
        loop = threads_new;
        stress_test = true;

        std::cout << "\nRunning stress test...";
    }

    // Kernel
    int times = 0;

    while (true)
    { // infinite loop for continuous tests
        MyClock::time_point t1 = MyClock::now();

#pragma omp parallel for
        for (int l = 0; l < loop; ++l)
        { // main loop
            do { // infinite loop when doing stress test
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
                    r0 = _mm256_addsub_ps(r2, r3);
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

                alignas(32) float mem[16];
                _mm256_store_ps(mem, r0);
                _mm256_store_ps(mem + 8, r1);
            } while (stress_test);
        }

        MySeconds time_span = std::chrono::duration_cast<MySeconds>(MyClock::now() - t1);
        ++times;

        std::cout << std::setprecision(6)
            << times << ": It took " << time_span.count()
            << " seconds to run " << loop << " loops.\n"
            << "    Average batch time (per loop) is "
            << std::setprecision(3)
            << std::chrono::duration_cast<MyMicroseconds>(time_span).count() / loop
            << " microseconds.\n"
            << std::setprecision(io_precision_origin);
    }

#ifdef _OPENMP
    omp_set_num_threads(threads_origin);
#endif
}

// Main
int main(int argc, char **argv)
{
    std::string input;

    // Set thread number
    int threads = 0;

    std::cout <<
        "Set the number of threads used for benchmark - default " + std::to_string(threads) + ".\n"
        "    0 means the number of physical processors' threads is used.\n"
        "    Leaving it blank implies the default setting.\n";

    while (true)
    {
        std::cout << "Your option: ";
        std::getline(std::cin, input);
        if (input == "") break;
        else threads = std::stoi(input);

        if (0) std::cout << "Invalid input! Try again.\n";
        else break;
    }

    std::cout << std::endl;

    // Set loop times
    int loop = 0x10000;

    std::cout <<
        "Set the number of loops used for benchmark - default " + std::to_string(loop) + ".\n"
        "    Use 0 for stress test (infinite loop).\n"
        "    Leaving it blank implies the default setting.\n";

    while (true)
    {
        std::cout << "Your option: ";
        std::getline(std::cin, input);
        if (input == "") break;
        else loop = std::stoi(input);

        if (loop < 0) std::cout << "Invalid input! Try again.\n";
        else break;
    }

    std::cout << std::endl;

    // Choose mode
    int mode = 1;

    std::cout <<
        "Choose mode - default " + std::to_string(mode) + ".\n"
        "    1: AVX operator test\n"
        "    Leaving it blank implies the default setting.\n";

    while (true)
    {
        std::cout << "Your option: ";
        std::getline(std::cin, input);
        if (input == "") break;
        else mode = std::stoi(input);

        if (mode < 1 || mode > 1) std::cout << "Invalid input! Try again.\n";
        else break;
    }

    std::cout << std::endl;

    // Benchmark
    switch (mode)
    {
    case 1:
        AVXOperatorTest(threads, loop);
        break;
    default:
        break;
    }

    return 0;
}
