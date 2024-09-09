package cuda

//#include <cuda.h>
import "C"

type CudaStream struct {
	stream C.CUstream
}
