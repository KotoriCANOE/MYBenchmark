#include "instruction_test.hpp"
#include <memory>

// Main
int main(int argc, char **argv)
{
    std::string input;

    // Set thread number
    int threads = 0;
#ifdef _OPENMP
    const int threads_origin = omp_get_max_threads();
    threads = threads_origin;
#endif

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
    int loop = 0x200;
#ifdef _OPENMP
    loop *= threads_origin;
#endif

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
    int mode = 3;

    std::cout <<
        "Choose mode - default " + std::to_string(mode) + ".\n"
        "    1: AVX operator test\n"
        "    2: AVX2+FMA operator test\n"
        "    3: AVX-512F operator test\n"
        "    Leaving it blank implies the default setting.\n";

    while (true)
    {
        std::cout << "Your option: ";
        std::getline(std::cin, input);
        if (input == "") break;
        else mode = std::stoi(input);

        if (mode < 1 || mode > 3) std::cout << "Invalid input! Try again.\n";
        else break;
    }

    std::cout << std::endl;

    // Choose type
    int type = 1;

    std::cout <<
        "Choose type - default " + std::to_string(type) + ".\n"
        "    1: FMA test (pure computing throughput)\n"
        "    2: FMA test (with memory read stress)\n"
        "    3: FMA test (with memory read+write stress)\n"
        "    4: Mixed test 1\n"
        "    5: Mixed test 2\n"
        "    Leaving it blank implies the default setting.\n";

    while (true)
    {
        std::cout << "Your option: ";
        std::getline(std::cin, input);
        if (input == "") break;
        else type = std::stoi(input);

        if (type < 1 || type > 5) std::cout << "Invalid input! Try again.\n";
        else break;
    }

    std::cout << std::endl;

    // Benchmark
    std::shared_ptr<InstructionTest> instT = nullptr;

    switch (mode)
    {
    case 1:
        instT = std::make_shared<AVXTest>();
        break;
    case 2:
        instT = std::make_shared<AVX2Test>();
        break;
    case 3:
        instT = std::make_shared<AVX512FTest>();
        break;
    default:
        break;
    }

    instT->threads = threads;
    instT->loop = loop;
    instT->type = type;
    instT->RunTest();

    return 0;
}
