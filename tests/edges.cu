//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <math.h>

// i = row, j = col
__device__ unsigned int getVal(unsigned char *arr, int i, int j, int width, int height) {
    if (i < 0 || i >= width || j < 0 || j >= height) {
        return 0;
    } else {
        return arr[j * width + i];
    }
}

extern "C" __global__ void borders(unsigned char *origImage, int width, int height, unsigned char *gradient, int imgSize) {
    // row
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // col
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = j * width + i;
    while (offset < imgSize) {
        unsigned int val1 = getVal(origImage, i - 1, j - 1, width, height);
        unsigned int val2 = getVal(origImage, i, j - 1, width, height);
        unsigned int val3 = getVal(origImage, i + 1, j - 1, width, height);
        unsigned int val4 = getVal(origImage, i - 1, j, width, height);
        unsigned int val5 = getVal(origImage, i + 1, j, width, height);
        unsigned int val6 = getVal(origImage, i - 1, j + 1, width, height);
        unsigned int val7 = getVal(origImage, i, j + 1, width, height);
        unsigned int val8 = getVal(origImage, i + 1, j + 1, width, height);

        int x = -val1 - 2 * val2 - val3 + val6 + 2 * val7 + val8;
        int y = val1 + 2 * val4 + val6 - val3 - 2 * val5 - val8;

        int _sqrt = sqrtf(x * x + y * y);
        // (_sqrt & -(_sqrt <= 255)) | (255 & -(_sqrt > 255));
        gradient[offset] = (_sqrt & -(_sqrt <= 255)) | (255 & -(_sqrt > 255));
        offset += blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    }
}