package cuda

//#include <cuda.h>
import "C"
import (
	"errors"
	"unsafe"
)

type Function struct {
	fun C.CUfunction
}

func (f *Function) AsKernel() (*Kernel, error) {
	return &Kernel{C.CUkernel(unsafe.Pointer(f.fun))}, nil
}

type Module struct {
	mod C.CUmodule
}

// LoadModule loads a CUDA module from a file
func LoadModule(path string) (*Module, error) {
	pathC := C.CString(path)
	defer C.free(unsafe.Pointer(pathC))
	var module C.CUmodule
	//path is a null terminated string
	stat := C.cuModuleLoad(&module, pathC)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Module{module}, nil
}

// LoadModuleData loads a CUDA module from a byte slice
func LoadModuleData(data []byte, isPtx bool) (*Module, error) {

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

	return &Module{module}, nil
}

func LoadModuleFatBin(data []byte) (*Module, error) {
	if len(data) == 0 {
		return nil, errors.New("data is empty")
	}

	var module C.CUmodule
	stat := C.cuModuleLoadFatBinary(&module, unsafe.Pointer(&data[0]))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Module{module}, nil
}

func (m *Module) Unload() error {
	stat := C.cuModuleUnload(C.CUmodule(m.mod))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (m *Module) GetFunctions() ([]*Function, error) {
	var count C.uint
	stat := C.cuModuleGetFunctionCount(&count, C.CUmodule(m.mod))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	functions := C.malloc(C.size_t(count) * C.sizeof_CUfunction)
	defer C.free(functions)

	stat = C.cuModuleEnumerateFunctions((*C.CUfunction)(functions), count, C.CUmodule(m.mod))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	functionsSlice := make([]*Function, count)
	for i := 0; i < int(count); i++ {
		functionsSlice[i] = &Function{C.CUfunction(unsafe.Pointer(uintptr(functions) + uintptr(i)*C.sizeof_CUfunction))}
	}

	return functionsSlice, nil
}

func (m *Module) GetFunction(name string) (*Function, error) {
	nameC := C.CString(name)
	defer C.free(unsafe.Pointer(nameC))
	var function C.CUfunction
	stat := C.cuModuleGetFunction(&function, C.CUmodule(m.mod), nameC)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Function{function}, nil
}

func (m *Module) GetGlobal(name string) (*MemAllocation, error) {
	nameC := C.CString(name)
	defer C.free(unsafe.Pointer(nameC))
	var mem C.CUdeviceptr
	var size C.size_t
	stat := C.cuModuleGetGlobal(&mem, &size, C.CUmodule(m.mod), nameC)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &MemAllocation{uintptr(mem), uint64(size), false}, nil
}
