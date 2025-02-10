#include "common.hpp"
#include "generator.hpp"
#include "callbacks.cuh"
#include <chrono>
// #include "helper_cuda.h"

using namespace std;

// const char *form = "%s;%d;%d\n";
const char *form = "%s;%f;\n";
// const char *header = "Func;Time;RunNum\n";
const char *header = "Func;Time;\n";

std::vector<generatorParam> get(dim3 block, dim3 grid); // definition for function contained in genereated file

void report(const char *func, int64_t time /*reported in nanoseconds*/)
{
    printf(form, func, time/1000.0); // microseconds
}

int main(int argc, char *argv[])
{
    dim3 grid(1, 1);
    dim3 block(32, 32);

    cout << header;

    runOnce(grid, block);
    log("Ran once");

    vector<generatorParam> params = get(block, grid);

    generatorState paramsState;

    auto gen = generator(params, paramsState);

    log("Starting tests");
    for (auto f = gen(); f != nullopt; f = gen())
    {
        (*f)();
    }
    log("Tests done");

    return 0;
}
