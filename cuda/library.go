package cuda

//#include <cuda.h>
import "C"
import (
	"unsafe"
)

// Represents a CUDA compiled library
type Library struct {
	lib C.CUlibrary
}

// JIT Option
type JitOptionName C.CUjit_option

// Library Option
type LibraryOptionName C.CUlibraryOption

// Helper struct for JIT options
type JitOption struct {
	Option JitOptionName
	Value  uint32
}

// Helper struct for Library options
type LibraryOption struct {
	Option LibraryOptionName
	Value  uint32
}

// Load a library from the specified path with options.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205
func LoadLibraryFromPath(path string, jit_options []JitOption, library_options []LibraryOption) (*Library, Result) {
	pathC := C.CString(path)
	defer C.free(unsafe.Pointer(pathC))

	var lib C.CUlibrary
	_, _, jitOptsAddr, jitValsAddr := parseJitOptions(jit_options)
	_, _, libOptsAddr, libValsAddr := parseLibraryOptions(library_options)
	stat := C.cuLibraryLoadFromFile(&lib, pathC, jitOptsAddr, jitValsAddr, C.uint(len(jit_options)),
		libOptsAddr, libValsAddr, C.uint(len(library_options)))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &Library{lib}, nil
}

// Load a library with specified code and options.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d
func LoadLibraryData(data []byte, jit_options []JitOption, library_options []LibraryOption) (*Library, Result) {
	if len(data) == 0 {
		return nil, ErrDataIsEmtpy
	}

	var lib C.CUlibrary
	_, _, jitOptsAddr, jitValsAddr := parseJitOptions(jit_options)
	_, _, libOptsAddr, libValsAddr := parseLibraryOptions(library_options)
	stat := C.cuLibraryLoadData(&lib, unsafe.Pointer(&data[0]), jitOptsAddr, jitValsAddr, C.uint(len(jit_options)),
		libOptsAddr, libValsAddr, C.uint(len(library_options)))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &Library{lib}, nil
}

// Unload the library.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914
func (lib *Library) Unload() Result {
	stat := C.cuLibraryUnload(lib.lib)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

// Get a kernel from the library by name.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a
func (lib *Library) GetKernel(name string) (*Kernel, Result) {
	nameC := C.CString(name)
	defer C.free(unsafe.Pointer(nameC))
	var kernel C.CUkernel
	stat := C.cuLibraryGetKernel(&kernel, lib.lib, nameC)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &Kernel{kernel}, nil
}

// Returns a pointer to managed memory.
// Currently not fully implemented.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1ga03f44378227ea68e6decd9d11c28fdf
func (lib *Library) GetManaged(name string) (*DeviceMemory, Result) {
	// TODO: Change the return type to a managed memory type
	return nil, ErrUnsupported
	nameC := C.CString(name)
	defer C.free(unsafe.Pointer(nameC))

	var mem C.CUdeviceptr
	var size C.size_t
	stat := C.cuLibraryGetManaged(&mem, &size, lib.lib, nameC)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &DeviceMemory{uintptr(mem), uint64(size), true}, nil
}

// Returns a pointer to a unified function.
// Currently not fully implemented.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1gb1b0ea992d64345562b694fdcd2c0334
func (lib *Library) GetUnifiedFunction(name string) (*Function, Result) {
	// TODO: Change the return type to a unified function type
	return nil, ErrUnsupported
	nameC := C.CString(name)
	defer C.free(unsafe.Pointer(nameC))
	var fun unsafe.Pointer
	stat := C.cuLibraryGetUnifiedFunction(&fun, lib.lib, nameC)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return nil, nil
}

// Returns a module.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g0d439597c77b64cf247de33f0609a5d8
func (lib *Library) GetModule() (*Module, Result) {
	var module C.CUmodule
	stat := C.cuLibraryGetModule(&module, lib.lib)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &Module{module}, nil
}

// Retrieve the kernel handles within the library.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1ga8ae2f42ab3a8fe789ac2dced8219608
func (lib *Library) GetKernels() ([]*Kernel, Result) {
	// TODO: think if we need to return kernel pointers
	var count C.uint
	stat := C.cuLibraryGetKernelCount(&count, lib.lib)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	kernels := make([]C.CUkernel, int(count))
	stat = C.cuLibraryEnumerateKernels(&kernels[0], count, lib.lib)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	res := make([]*Kernel, int(count))
	for i := 0; i < int(count); i++ {
		res[i] = &Kernel{kernels[i]}
	}
	return res, nil
}

// Returns a global device pointer.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g98708b50c11bc1c0addd6ecab96ae4ab
func (lib *Library) GetGlobal(name string) (*DeviceMemory, Result) {
	nameC := C.CString(name)
	defer C.free(unsafe.Pointer(nameC))
	var mem C.CUdeviceptr
	var size C.size_t
	stat := C.cuLibraryGetGlobal(&mem, &size, lib.lib, nameC)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &DeviceMemory{uintptr(mem), uint64(size), false}, nil
}

// Returns the native pointer of the library.
func (lib *Library) NativePointer() uintptr {
	return uintptr(unsafe.Pointer(lib.lib))
}

func parseJitOptions(options []JitOption) (opts []C.CUjit_option, vals []C.uint,
	optsAddr *C.CUjit_option, valsAddr *unsafe.Pointer) {
	opts = make([]C.CUjit_option, len(options))
	vals = make([]C.uint, len(options))

	for i, opt := range options {
		opts[i] = C.CUjit_option(opt.Option)
		vals[i] = C.uint(opt.Value)
	}

	if len(options) > 0 {
		optsAddr = &opts[0]
		valsAddr = (*unsafe.Pointer)(unsafe.Pointer(&vals[0]))
	}

	return opts, vals, optsAddr, valsAddr
}

func parseLibraryOptions(options []LibraryOption) (opts []C.CUlibraryOption, vals []C.uint,
	optsAddr *C.CUlibraryOption, valsAddr *unsafe.Pointer) {
	opts = make([]C.CUlibraryOption, len(options))
	vals = make([]C.uint, len(options))

	for i, opt := range options {
		opts[i] = C.CUlibraryOption(opt.Option)
		vals[i] = C.uint(opt.Value)
	}

	if len(options) > 0 {
		optsAddr = &opts[0]
		valsAddr = (*unsafe.Pointer)(unsafe.Pointer(&vals[0]))
	}

	return opts, vals, optsAddr, valsAddr
}

const (
	CU_JIT_MAX_REGISTERS                    JitOptionName = 0
	CU_JIT_THREADS_PER_BLOCK                JitOptionName = 1
	CU_JIT_WALL_TIME                        JitOptionName = 2
	CU_JIT_INFO_LOG_BUFFER                  JitOptionName = 3
	CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES       JitOptionName = 4
	CU_JIT_ERROR_LOG_BUFFER                 JitOptionName = 5
	CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES      JitOptionName = 6
	CU_JIT_OPTIMIZATION_LEVEL               JitOptionName = 7
	CU_JIT_TARGET_FROM_CUCONTEXT            JitOptionName = 8
	CU_JIT_TARGET                           JitOptionName = 9
	CU_JIT_FALLBACK_STRATEGY                JitOptionName = 10
	CU_JIT_GENERATE_DEBUG_INFO              JitOptionName = 11
	CU_JIT_LOG_VERBOSE                      JitOptionName = 12
	CU_JIT_GENERATE_LINE_INFO               JitOptionName = 13
	CU_JIT_CACHE_MODE                       JitOptionName = 14
	CU_JIT_NEW_SM3X_OPT                     JitOptionName = 15
	CU_JIT_FAST_COMPILE                     JitOptionName = 16
	CU_JIT_GLOBAL_SYMBOL_NAMES              JitOptionName = 17
	CU_JIT_GLOBAL_SYMBOL_ADDRESSES          JitOptionName = 18
	CU_JIT_GLOBAL_SYMBOL_COUNT              JitOptionName = 19
	CU_JIT_LTO                              JitOptionName = 20
	CU_JIT_FTZ                              JitOptionName = 21
	CU_JIT_PREC_DIV                         JitOptionName = 22
	CU_JIT_PREC_SQRT                        JitOptionName = 23
	CU_JIT_FMA                              JitOptionName = 24
	CU_JIT_REFERENCED_KERNEL_NAMES          JitOptionName = 25
	CU_JIT_REFERENCED_KERNEL_COUNT          JitOptionName = 26
	CU_JIT_REFERENCED_VARIABLE_NAMES        JitOptionName = 27
	CU_JIT_REFERENCED_VARIABLE_COUNT        JitOptionName = 28
	CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES JitOptionName = 29
	CU_JIT_POSITION_INDEPENDENT_CODE        JitOptionName = 30
	CU_JIT_MIN_CTA_PER_SM                   JitOptionName = 31
	CU_JIT_MAX_THREADS_PER_BLOCK            JitOptionName = 32
	CU_JIT_OVERRIDE_DIRECTIVE_VALUES        JitOptionName = 33

	//see https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a1cdb7004bb8a24f1342de9004add23
	CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE LibraryOptionName = 0
	CU_LIBRARY_BINARY_IS_PRESERVED                    LibraryOptionName = 1
)
