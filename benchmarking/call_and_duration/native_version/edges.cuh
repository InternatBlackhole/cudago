#ifndef _EDGES_H_
#define _EDGES_H_

extern "C" __global__ void borders(unsigned char *origImage, int width, int height, unsigned char *gradient, int imgSize);

#endif