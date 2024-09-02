package cuda

//#include <cuda.h>
import "C"
import (
	"errors"
	"unsafe"
)

type CudaFunction C.CUfunction
type CudaModule struct {
	mod C.CUmodule
}

// LoadModule loads a CUDA module from a file
func LoadModule(path string) (*CudaModule, error) {
	var module C.CUmodule
	//path is a null terminated string
	stat := C.cuModuleLoad(&module, C.CString(path))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &CudaModule{module}, nil
}

// LoadModuleData loads a CUDA module from a byte slice
func LoadModuleData(data []byte, isPtx bool) (*CudaModule, error) {

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
		return nil, NewCudaError(uint32(stat))
	}

	return &CudaModule{module}, nil
}

func LoadModuleFatBin(data []byte) (*CudaModule, error) {
	if len(data) == 0 {
		return nil, errors.New("data is empty")
	}

	var module C.CUmodule
	stat := C.cuModuleLoadFatBinary(&module, unsafe.Pointer(&data[0]))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &CudaModule{module}, nil
}

func (m *CudaModule) Unload() error {
	stat := C.cuModuleUnload(C.CUmodule(m.mod))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (m *CudaModule) GetFunction(name string) (CudaFunction, error) {
	var function C.CUfunction
	stat := C.cuModuleGetFunction(&function, C.CUmodule(m.mod), C.CString(name))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return CudaFunction(function), nil
}
