package cuda

//#include <cuda.h>
import "C"
import (
	"errors"
	"unsafe"
)

type CudaModule C.CUmodule

// LoadModule loads a CUDA module from a file
func LoadModule(path string) (CudaModule, error) {
	var module C.CUmodule
	//path is a null terminated string
	stat := C.cuModuleLoad(&module, C.CString(path))

	if stat != C.CUDA_SUCCESS {
		return CudaModule(module), errors.New(ResultMap[cudaResult(stat)])
	}

	return CudaModule(module), nil
}

// LoadModuleData loads a CUDA module from a byte slice
func LoadModuleData(data []byte, isPtx bool) (CudaModule, error) {

	if len(data) == 0 {
		return nil, errors.New("data is empty")
	}

	if isPtx && data[len(data)-1] != 0 {
		data = append(data, 0)
	}

	var module C.CUmodule
	//data if PTX has to be null terminated
	stat := C.cuModuleLoadData(&module, unsafe.Pointer(&data[0]))

	if stat != C.CUDA_SUCCESS {
		return nil, errors.New(ResultMap[cudaResult(stat)])
	}

	return CudaModule(module), nil
}
