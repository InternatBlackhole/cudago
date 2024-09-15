package cuda

//#include <cuda.h>
import "C"
import (
	"unsafe"
)

type Module struct {
	mod C.CUmodule
}

type LoadingMode int32

// LoadModule loads a CUDA module from a file
func LoadModule(path string) (*Module, Result) {
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

// LoadModuleData loads a CUDA module from a byte slice, the PTX data has to have a null terminator
func LoadModuleData(data []byte) (*Module, Result) {
	return LoadModuleDataEx(data, nil)
}

func LoadModuleDataEx(data []byte, options []JitOption) (*Module, Result) {
	if len(data) == 0 {
		return nil, ErrDataIsEmtpy
	}

	var module C.CUmodule

	_, _, optionsAddr, valuesAddr := parseJitOptions(options)

	stat := C.cuModuleLoadDataEx(&module, unsafe.Pointer(&data[0]),
		C.uint(len(options)), optionsAddr, valuesAddr)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Module{module}, nil
}

func LoadModuleFatBin(data []byte) (*Module, Result) {
	if len(data) == 0 {
		return nil, ErrDataIsEmtpy
	}

	var module C.CUmodule
	stat := C.cuModuleLoadFatBinary(&module, unsafe.Pointer(&data[0]))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Module{module}, nil
}

func (m *Module) Unload() Result {
	stat := C.cuModuleUnload(C.CUmodule(m.mod))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (m *Module) GetFunctions() ([]*Function, Result) {
	var count C.uint
	stat := C.cuModuleGetFunctionCount(&count, C.CUmodule(m.mod))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	//functions := C.malloc(C.size_t(count) * C.sizeof_CUfunction)
	//defer C.free(functions)

	functions := make([]C.CUfunction, count)

	stat = C.cuModuleEnumerateFunctions(&functions[0], count, C.CUmodule(m.mod))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	functionsSlice := make([]*Function, count)
	for i := 0; i < int(count); i++ {
		functionsSlice[i] = &Function{functions[i]}
	}

	return functionsSlice, nil
}

func (m *Module) GetFunction(name string) (*Function, Result) {
	nameC := C.CString(name)
	defer C.free(unsafe.Pointer(nameC))
	var function C.CUfunction
	stat := C.cuModuleGetFunction(&function, C.CUmodule(m.mod), nameC)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Function{function}, nil
}

func (m *Module) GetGlobal(name string) (*DeviceMemory, Result) {
	nameC := C.CString(name)
	defer C.free(unsafe.Pointer(nameC))
	var mem C.CUdeviceptr
	var size C.size_t
	stat := C.cuModuleGetGlobal(&mem, &size, C.CUmodule(m.mod), nameC)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &DeviceMemory{uintptr(mem), uint64(size), false}, nil
}

func (m *Module) NativePointer() uintptr {
	return uintptr(unsafe.Pointer(m.mod))
}

func GetModuleLoadingMode() (LoadingMode, Result) {
	var mode C.CUmoduleLoadingMode
	stat := C.cuModuleGetLoadingMode(&mode)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return LoadingMode(mode), nil
}

const (
	CU_MODULE_EAGER_LOADING LoadingMode = 0x1 //Lazy Kernel Loading is not enabled
	CU_MODULE_LAZY_LOADING  LoadingMode = 0x2 //Lazy Kernel Loading is enabled
)
