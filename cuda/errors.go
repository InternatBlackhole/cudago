package cuda

//#include <cuda.h>
import "C"

// TODO: Implement a general error that is not specific to CUDA

type cudaResult struct {
	res C.CUresult
}

func (r cudaResult) String() string {
	return r.Error()
}

func (r cudaResult) Error() string {
	return r.ErrorName() + ": " + r.ErrorString()
}

func (r cudaResult) Code() uint32 {
	return uint32(r.res)
}

func (r cudaResult) ErrorString() string {
	var str *C.char
	err := C.cuGetErrorString(r.res, &str)
	if err != C.CUDA_SUCCESS {
		return "CUDA_ERROR_INVALID_VALUE"
	}
	return C.GoString(str)
}

func (r cudaResult) ErrorName() string {
	var str *C.char
	err := C.cuGetErrorName(r.res, &str)
	if err != C.CUDA_SUCCESS {
		return "CUDA_ERROR_INVALID_VALUE"
	}
	return C.GoString(str)
}

func NewCudaError(err uint32) error {
	return cudaResult{C.CUresult(err)}
}
