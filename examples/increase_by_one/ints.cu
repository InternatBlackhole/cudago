//#include <cuda.h>
#ifdef __cplusplus
extern "C" {
#endif

__global__ void addToAll(int *orig, int toAdd, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int org = orig[idx];
        orig[idx] = org + toAdd;
        printf("idx = %d, orig = %d, newOrig = %d\n", idx, org, orig[idx]);
    }
}

#ifdef __cplusplus
}
#endif