#ifndef _INTS_H_
#define _INTS_H_

#ifdef __cplusplus
extern "C" {
#endif

__global__ void addToAll(int *orig, int toAdd, int size);

#ifdef __cplusplus
}
#endif

#endif