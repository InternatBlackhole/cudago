#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#ifdef __cplusplus
extern "C" {
#endif

__global__ void bitonicSortStart(int *a, int len);
__global__ void bitonicSortMiddle(int *a, int len, int k, int j);
__global__ void bitonicSortFinish(int *a, int len, int k);

#ifdef __cplusplus
}
#endif

#endif // BITONIC_SORT_H