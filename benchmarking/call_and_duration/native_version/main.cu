#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <paths.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <string>

#include "helper_cuda.h"

#include "edges.cuh"
#include "ints.cuh"
#include "bitonic-sort.cuh"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#define COLOR_CHANNELS 1
int validate_file(char *filename)
{
    struct stat buffer = {0};
    int ret = stat(filename, &buffer);
    if (ret)
    {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }
    return 0;
}

#define log(msg) std::cerr << msg << std::endl
#define logf(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__)

void borders(const char *inFile, const char *outFile, const char *baseName);
void sort(int32_t *data, int dataLen, int elemSize, int numThreads);
void increase(int32_t *data, int dataLen, int elemSize, int numThreads, int by);

const char *reportFormat = "%s;\"%s\";%f\n";
const char *reportHeader = "Operation;Image;Time\n";

#define dur(start, end) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
using timer = std::chrono::high_resolution_clock;

void report(const char *func, const char* image, int64_t time)
{
    printf(reportFormat, func, image, time/1000.0); // microseconds
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        logf("Usage: %s <out_dir> <images...>\n", argv[0]);
        exit(1);
    }

    std::string outDir(argv[1]);
    bool v = false;
    if (outDir == "/dev/null")
    {
        log("VOIDING IMAGES!!!!\n");
        v = true;
    }
    else
    {
        // validates directory
        validate_file(argv[1]);
    }

    printf(reportHeader);

    log("Starting border recognition...");

    for (int i = 2; i < argc; i++)
    {
        std::string fullname = std::string(argv[i]);
        size_t lastindex = fullname.find_last_of(".");
        std::string noext = fullname.substr(0, lastindex);

        std::string filename(basename(fullname.c_str()));

        std::string gpuOut = v ? "/dev/null" : std::string(argv[1]) + '/' + filename + ".png";

        borders(argv[i], gpuOut.c_str(), filename.c_str());
        std::cerr << std::endl;
    }

    log("Border recognition done.");
    log("Starting sorting and increase...");

    int numThreads = 128;

    // 8192 int32s = 32KB, 2^25 ints = 128MB
    for (size_t size = 1 << 13; size <= 1 << 25; size <<= 1)
    {
        std::cerr << "Size: " << size << std::endl;
        int32_t *data = 0;
        size_t elemSize = sizeof(int32_t);
        checkCudaErrors(cudaHostAlloc((void **)&data, size * elemSize, cudaHostAllocPortable));
        for (size_t i = 0; i < size; i++)
        {
            data[i] = size - i;
        }

        int32_t* d_arr = 0;
        checkCudaErrors(cudaMalloc((void **)&d_arr, size * elemSize));
        
        checkCudaErrors(cudaMemcpy(d_arr, data, size * elemSize, cudaMemcpyHostToDevice));

        increase(d_arr, size, elemSize, numThreads, 10);
        sort(d_arr, size, elemSize, numThreads);

        checkCudaErrors(cudaMemcpy(data, d_arr, size * elemSize, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_arr));
        checkCudaErrors(cudaFreeHost(data));
    }

    checkCudaErrors(cudaDeviceSynchronize());

    log("Sorting and increase done.");

    return 0;
}

void borders(const char *inFile, const char *outFile, const char *baseName)
{
    int block_size = 32;
    int width, height, bpp;
    unsigned char *origImage = stbi_load(inFile, &width, &height, &bpp, COLOR_CHANNELS);
    if (origImage == NULL)
    {
        logf("Error loading image %s\n", baseName);
        return;
    }

    int size = width * height * COLOR_CHANNELS * sizeof(unsigned char);
    unsigned char *d_gray_image;
    unsigned char *d_grad;
    checkCudaErrors(cudaHostRegister(origImage, size, cudaHostRegisterMapped));
    checkCudaErrors(cudaMalloc((void **)&d_gray_image, size));
    checkCudaErrors(cudaMalloc((void **)&d_grad, size));

    dim3 dimBlock(block_size, block_size, 1);
    dim3 dimGrid(ceil(width / float(block_size)), ceil(height / float(block_size)), 1);

    logf("CUDA config for %s: block_size: %d, grid_size: (x: %d, y: %d, z: %d)\n", baseName, block_size, dimGrid.x, dimGrid.y, dimGrid.z);

    cudaEvent_t start, stop, edgesKernelStart, edgesKernelEnd;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&edgesKernelStart);
    cudaEventCreate(&edgesKernelEnd);

    log("GPU start");
    // unsigned char *finalImage = (unsigned char *)malloc(size);
    unsigned char *finalImage = 0;
    checkCudaErrors(cudaHostAlloc((void **)&finalImage, size, cudaHostAllocMapped));

    checkCudaErrors(cudaEventRecord(edgesKernelStart));

    logf("Copy start %s to device...\n", baseName);
    checkCudaErrors(cudaMemcpy(d_gray_image, origImage, size, cudaMemcpyHostToDevice));
    logf("Copy done %s to device...\n", baseName);

    checkCudaErrors(cudaEventRecord(edgesKernelStart));

    logf("Kernel call start %s...\n", baseName);
    auto startK = timer::now();
    borders<<<dimGrid, dimBlock>>>(d_gray_image, width, height, d_grad, size);
    auto endK = timer::now();
    logf("Kernel call done %s...\n", baseName);

    checkCudaErrors(cudaEventRecord(edgesKernelEnd));

    // checkCudaErrors(cudaEventSynchronize(edgesKernelEnd))

    logf("Copy start %s from device...\n", baseName);
    checkCudaErrors(cudaMemcpy(finalImage, d_grad, size, cudaMemcpyDeviceToHost));
    logf("Copy done %s from device...\n", baseName);

    checkCudaErrors(cudaEventRecord(stop));

    checkCudaErrors(cudaEventSynchronize(stop));

    float gpu_milliseconds = 0;
    // checkCudaErrors(cudaEventElapsedTime(&gpu_milliseconds, start, stop));
    checkCudaErrors(cudaEventElapsedTime(&gpu_milliseconds, edgesKernelStart, edgesKernelEnd));

    //const std::chrono::duration<double, std::milli> duration = endK - startK;

    report("GoStartToEndKernelCall", baseName, dur(startK, endK));
    printf(reportFormat, "KernelCall", baseName, gpu_milliseconds);

    stbi_write_png(outFile, width, height, COLOR_CHANNELS, finalImage, width * COLOR_CHANNELS);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(edgesKernelStart);
    cudaEventDestroy(edgesKernelEnd);
    cudaFreeHost(finalImage);
    cudaHostUnregister(origImage);
    cudaFree(d_gray_image);
    cudaFree(d_grad);
    stbi_image_free(origImage);
}

void sort(int32_t *data, int dataLen, int elemSize, int numThreads) {
    int numBlocks = (dataLen/2 - 1) / numThreads + 1;
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    dim3 grid(numBlocks, 1, 1);
    dim3 block(numThreads, 1, 1);
    int bytesLocalMemory = 2 * block.x * elemSize;

    checkCudaErrors(cudaEventRecord(start));

    auto startT = timer::now();
    bitonicSortStart<<<grid, block, bytesLocalMemory>>>(data, dataLen);
    auto endT = timer::now();
    report("Sort_GoStartKernelCall", "", dur(startT, endT));
    for (size_t k = 4 * block.x; k <= dataLen; k <<= 1)
    {
        for (size_t j = k / 2; j >= 2*block.x; j >>= 1) {
            startT = timer::now();
            bitonicSortMiddle<<<grid, block, bytesLocalMemory>>>(data, dataLen, k, j);
            endT = timer::now();
            report("Sort_GoMiddleKernelCall", "", dur(startT, endT));
        }
        startT = timer::now();
        bitonicSortFinish<<<grid, block, bytesLocalMemory>>>(data, dataLen, k);
        endT = timer::now();
        report("Sort_GoFinishKernelCall", "", dur(startT, endT));
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float gpu_milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&gpu_milliseconds, start, stop));

    printf(reportFormat, "Sort_KernelCall", "", gpu_milliseconds);
    
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
}

void increase(int32_t *data, int dataLen, int elemSize, int numThreads, int by) {
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    dim3 grid(ceil(dataLen / float(numThreads)), 1, 1);
    dim3 block(numThreads, 1, 1);

    checkCudaErrors(cudaEventRecord(start));

    auto startT = timer::now();
    addToAll<<<grid, block>>>(data, dataLen, by);
    auto endT = timer::now();
    checkCudaErrors(cudaEventRecord(stop));

    checkCudaErrors(cudaEventSynchronize(stop));

    float gpu_milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&gpu_milliseconds, start, stop));

    report("AddToAll_GoStartToEndKernelCall", "", dur(startT, endT));
    printf(reportFormat, "AddToAll_KernelCall", "", gpu_milliseconds);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
}