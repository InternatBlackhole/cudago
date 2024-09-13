// računanje razlike vektorjev
// 		argumenti: število blokov, število niti in dolžina vektorjev
// 		elementi vektorjev so inicializirani naključno
// uporaba enotnega pomnilnika (v tem file se ne)


__global__ void vectorSubtract(float *c, const float *a, const float *b, int len) {
	// določimo globalni indeks elementov
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	// če je niti manj kot je dolžina vektorjev, morajo nekatere narediti več elementov
	while (gid < len) {
		c[gid] = a[gid] - b[gid];
		gid += gridDim.x * blockDim.x;
	}
}