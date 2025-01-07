package cuda

//#include <cuda.h>
import "C"
import (
	"unsafe"
)

// Represents a CUDA module.
type Module struct {
	mod C.CUmodule
}

// Represents a module loading mode.
type LoadingMode int32

// Loads a CUDA module from a file.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3
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

// Loads a CUDA module from a byte slice.
// PTX data has to have a null terminator.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b
func LoadModuleData(data []byte) (*Module, Result) {
	return LoadModuleDataEx(data, nil)
}

// Loads a CUDA module from a byte slice with options.
// PTX data has to have a null terminator.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12
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

// Loads a module data in fatbin.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g13a2292b6819f8f86127768334436c3b
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

// Unloads the module.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b
func (m *Module) Unload() Result {
	stat := C.cuModuleUnload(C.CUmodule(m.mod))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Returns functions within the module.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g6bdb22a7d9cacf7df5bda2a18082ec50
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

// Returns a function by name.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf
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

// Returns a global pointer from module by name.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab
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

// Returns the native pointer of the module.
func (m *Module) NativePointer() uintptr {
	return uintptr(unsafe.Pointer(m.mod))
}

// Query loading mode.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g96de378a738ec46d9277c9c9df8f6fd6
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
