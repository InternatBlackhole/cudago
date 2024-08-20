package cuda_runtime

//#include <cuda_runtime.h>
import "C"

type CudaError struct {
	Code uint32 // cudaError_t
}

func NewCudaError(code uint32) CudaError {
	return CudaError{code}
}

func (e CudaError) Error() string {
	return e.CodeName() + ":\n" + e.CodeString()
}

func (e CudaError) CodeName() string {
	return CudaErrorName(e.Code)
}

func (e CudaError) CodeString() string {
	return CudaErrorString(e.Code)
}

func CudaErrorName(err uint32) string {
	return C.GoString(C.cudaGetErrorName(C.cudaError_t(err)))
}

func CudaErrorString(err uint32) string {
	return C.GoString(C.cudaGetErrorString(C.cudaError_t(err)))
}

func LastCudaError() CudaError {
	code := uint32(C.cudaGetLastError())
	return CudaError{code}
}

func PeekLastCudaError() CudaError {
	code := uint32(C.cudaPeekAtLastError())
	return CudaError{code}
}
