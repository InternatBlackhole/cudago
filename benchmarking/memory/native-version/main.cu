#include "common.hpp"
#include "generator.hpp"
#include <chrono>
#include "helper_cuda.h"

#define dur(start, end) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()

using namespace std;

using timer = std::chrono::high_resolution_clock;

const char *form = "%s;%f\n";
const char *header = "Operation;Time\n";
void testNoReg(int len);
void testYesReg(int len);

void report(const char *func, int64_t time)
{
    printf(form, func, time/1000.0); // microseconds
}

int main(int argc, char *argv[])
{
    cout << header;

    //the first any memory allocation is slow, so we do it here
    //why is it like that? I assume some sort of lazy linking or initialization but idk
    int* addr = 0;
    checkCudaErrors(cudaHostAlloc(&addr, 1, cudaHostAllocPortable));
    checkCudaErrors(cudaFreeHost(addr));


    generatorState paramsState;

    std::vector<generatorParam> params = {
        {"CopyNoReg", 10, [](){ testNoReg(100000000); }}, // 100M == 800MB
        {"CopyYesReg", 10, [](){ testYesReg(100000000); }}, // 100M == 800MB
    };

    auto gen = generator(params, paramsState);

    log("Starting tests");
    for (auto f = gen(); f != nullopt; f = gen())
    {
        (*f)();
    }
    log("Tests done");

    return 0;
}

void testNoReg(int len) {
    int64_t* arr = 0;
    uint64_t arrBytes = len * sizeof(int64_t);

    timer::time_point start = timer::now();
    arr = (int64_t*)malloc(arrBytes);
    timer::time_point end = timer::now();
    if (arr == 0) {
        log("Failed to allocate memory");
        return;
    }
    report("HostMallocNormal", dur(start, end));

    for (uint64_t i = 0; i < len; i++) {
        arr[i] = i;
    }

    cudaError_t err;

    int64_t* d_arr = 0;
    start = timer::now();
    err = cudaMalloc((void**)&d_arr, arrBytes);
    end = timer::now();
    checkCudaErrors(err);
    report("DevMallocNoReg", dur(start, end));

    start = timer::now();
    err = cudaMemcpy(d_arr, arr, arrBytes, cudaMemcpyHostToDevice);
    end = timer::now();
    checkCudaErrors(err);
    report("MemcpyToDeviceNoReg", dur(start, end));

    start = timer::now();
    err = cudaMemcpy(arr, d_arr, arrBytes, cudaMemcpyDeviceToHost);
    end = timer::now();
    checkCudaErrors(err);
    report("MemcpyFromDeviceNoReg", dur(start, end));

    checkCudaErrors(cudaFree(d_arr));
    free(arr);
}

void testYesReg(int len) {
    int64_t* arr = 0;
    uint64_t arrBytes = len * sizeof(int64_t);
    cudaError_t err;

    timer::time_point start = timer::now();
    err = cudaHostAlloc(&arr, arrBytes, cudaHostAllocPortable);
    timer::time_point end = timer::now();
    checkCudaErrors(err);
    report("HostMallocCUDA", dur(start, end));

    for (uint64_t i = 0; i < len; i++) {
        arr[i] = i;
    }

    int64_t* d_arr = 0;
    start = timer::now();
    err = cudaMalloc((void**)&d_arr, arrBytes);
    end = timer::now();
    checkCudaErrors(err);
    report("DevMallocYesReg", dur(start, end));

    start = timer::now();
    err = cudaMemcpy(d_arr, arr, arrBytes, cudaMemcpyHostToDevice);
    end = timer::now();
    checkCudaErrors(err);
    report("MemcpyToDevYesReg", dur(start, end));

    start = timer::now();
    err = cudaMemcpy(arr, d_arr, arrBytes, cudaMemcpyDeviceToHost);
    end = timer::now();
    checkCudaErrors(err);
    report("MemcpyFromDevYesReg", dur(start, end));

    checkCudaErrors(cudaFree(d_arr));
    checkCudaErrors(cudaFreeHost(arr));
}