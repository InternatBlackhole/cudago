package cuda

//#include <cuda.h>
import "C"
import (
	"errors"
	"unsafe"
)

type CudaLibrary struct {
	lib C.CUlibrary
}

type JitOptionName C.CUjit_option
type LibraryOptionName C.CUlibraryOption

type JitOption struct {
	Option JitOptionName
	Value  uint32
}

type LibraryOption struct {
	Option LibraryOptionName
	Value  uint32
}

func LoadLibraryFromPath(path string, jit_options []JitOption, library_options []LibraryOption) (*CudaLibrary, error) {

	/*jitOptions := C.malloc(C.size_t(len(jit_options)) * C.size_t(unsafe.Sizeof(C.CUjit_option(0))))
	defer C.free(jitOptions)

	for i, opt := range jit_options {
		*(*C.CUjit_option)(unsafe.Pointer(uintptr(jitOptions) + uintptr(i)*unsafe.Sizeof(C.CUjit_option(0)))) = C.CUjit_option(opt.Option)
	}

	jitValues := C.malloc(C.size_t(len(jit_options)) * C.size_t(unsafe.Sizeof(C.uint(0))))
	defer C.free(jitValues)

	for i, opt := range jit_options {
		*(*C.uint)(unsafe.Pointer(uintptr(jitValues) + uintptr(i)*unsafe.Sizeof(C.uint(0)))) = C.uint(opt.Value)
	}

	libraryOptions := C.malloc(C.size_t(len(library_options)) * C.size_t(unsafe.Sizeof(C.CUlibraryOption(0))))
	defer C.free(libraryOptions)

	for i, opt := range library_options {
		*(*C.CUlibraryOption)(unsafe.Pointer(uintptr(libraryOptions) + uintptr(i)*unsafe.Sizeof(C.CUlibraryOption(0)))) = C.CUlibraryOption(opt.Option)
	}

	libraryValues := C.malloc(C.size_t(len(library_options)) * C.size_t(unsafe.Sizeof(C.uint(0))))
	defer C.free(libraryValues)

	for i, opt := range library_options {
		*(*C.uint)(unsafe.Pointer(uintptr(libraryValues) + uintptr(i)*unsafe.Sizeof(C.uint(0)))) = C.uint(opt.Value)
	}*/

	fun := func(jitOpts unsafe.Pointer, jitOptsVals *unsafe.Pointer, numJitOpt C.uint,
		libOpts unsafe.Pointer, libOptsVals *unsafe.Pointer, libOptNum C.uint) (*CudaLibrary, error) {

		var lib C.CUlibrary
		stat := C.cuLibraryLoadFromFile(&lib, C.CString(path), (*C.CUjit_option)(jitOpts), jitOptsVals, numJitOpt,
			(*C.CUlibraryOption)(libOpts), libOptsVals, libOptNum)

		if stat != C.CUDA_SUCCESS {
			return nil, NewCudaError(uint32(stat))
		}
		return &CudaLibrary{lib}, nil
	}

	/*var lib C.CUlibrary
	stat := C.cuLibraryLoadFromFile(&lib, C.CString(path), (*C.CUjit_option)(jitOptions), &jitValues, C.uint(len(jit_options)),
		(*C.CUlibraryOption)(libraryOptions), &libraryValues, C.uint(len(library_options)))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}*/

	//return &CudaLibrary{lib}, nil
	return internalLoad(jit_options, library_options, fun)
}

func LoadLibraryData(data []byte, jit_options []JitOption, library_options []LibraryOption) (*CudaLibrary, error) {

	if len(data) == 0 {
		return nil, errors.New("data is empty")
	}

	// i love breaking DRY /s
	// but seriously, figure out a way to cast to something that you can assign to

	/*jitOptions := C.malloc(C.size_t(len(jit_options)) * C.size_t(unsafe.Sizeof(C.CUjit_option(0))))
	defer C.free(jitOptions)

	for i, opt := range jit_options {
		*(*C.CUjit_option)(unsafe.Pointer(uintptr(jitOptions) + uintptr(i)*unsafe.Sizeof(C.CUjit_option(0)))) = C.CUjit_option(opt.Option)
	}

	jitValues := C.malloc(C.size_t(len(jit_options)) * C.size_t(unsafe.Sizeof(C.uint(0))))
	defer C.free(jitValues)

	for i, opt := range jit_options {
		*(*C.uint)(unsafe.Pointer(uintptr(jitValues) + uintptr(i)*unsafe.Sizeof(C.uint(0)))) = C.uint(opt.Value)
	}

	libraryOptions := C.malloc(C.size_t(len(library_options)) * C.size_t(unsafe.Sizeof(C.CUlibraryOption(0))))
	defer C.free(libraryOptions)

	for i, opt := range library_options {
		*(*C.CUlibraryOption)(unsafe.Pointer(uintptr(libraryOptions) + uintptr(i)*unsafe.Sizeof(C.CUlibraryOption(0)))) = C.CUlibraryOption(opt.Option)
	}

	libraryValues := C.malloc(C.size_t(len(library_options)) * C.size_t(unsafe.Sizeof(C.uint(0))))
	defer C.free(libraryValues)

	for i, opt := range library_options {
		*(*C.uint)(unsafe.Pointer(uintptr(libraryValues) + uintptr(i)*unsafe.Sizeof(C.uint(0)))) = C.uint(opt.Value)
	}*/

	fun := func(jitOpts unsafe.Pointer, jitOptsVals *unsafe.Pointer, numJitOpt C.uint,
		libOpts unsafe.Pointer, libOptsVals *unsafe.Pointer, libOptNum C.uint) (*CudaLibrary, error) {
		var lib C.CUlibrary
		stat := C.cuLibraryLoadData(&lib, unsafe.Pointer(&data[0]), (*C.CUjit_option)(jitOpts), jitOptsVals, numJitOpt,
			(*C.CUlibraryOption)(libOpts), libOptsVals, libOptNum)

		if stat != C.CUDA_SUCCESS {
			return nil, NewCudaError(uint32(stat))
		}
		return &CudaLibrary{lib}, nil
	}

	/*var lib C.CUlibrary
	stat := C.cuLibraryLoadData(&lib, unsafe.Pointer(&data[0]), (*C.CUjit_option)(jitOptions), &jitValues, C.uint(len(jit_options)),
		(*C.CUlibraryOption)(libraryOptions), &libraryValues, C.uint(len(library_options)))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}*/

	//return &CudaLibrary{lib}, nil
	return internalLoad(jit_options, library_options, fun)
}

type internalFunc func(jitOpts unsafe.Pointer, jitOptsVals *unsafe.Pointer, numJitOpt C.uint,
	libOpts unsafe.Pointer, libOptsVals *unsafe.Pointer, libOptNum C.uint) (*CudaLibrary, error)

func internalLoad(jit_options []JitOption, library_options []LibraryOption, fun internalFunc) (*CudaLibrary, error) {
	jitOptions := C.malloc(C.size_t(len(jit_options)) * C.size_t(unsafe.Sizeof(C.CUjit_option(0))))
	defer C.free(jitOptions)

	for i, opt := range jit_options {
		*(*C.CUjit_option)(unsafe.Pointer(uintptr(jitOptions) + uintptr(i)*unsafe.Sizeof(C.CUjit_option(0)))) = C.CUjit_option(opt.Option)
	}

	jitValues := C.malloc(C.size_t(len(jit_options)) * C.size_t(unsafe.Sizeof(C.uint(0))))
	defer C.free(jitValues)

	for i, opt := range jit_options {
		*(*C.uint)(unsafe.Pointer(uintptr(jitValues) + uintptr(i)*unsafe.Sizeof(C.uint(0)))) = C.uint(opt.Value)
	}

	libraryOptions := C.malloc(C.size_t(len(library_options)) * C.size_t(unsafe.Sizeof(C.CUlibraryOption(0))))
	defer C.free(libraryOptions)

	for i, opt := range library_options {
		*(*C.CUlibraryOption)(unsafe.Pointer(uintptr(libraryOptions) + uintptr(i)*unsafe.Sizeof(C.CUlibraryOption(0)))) = C.CUlibraryOption(opt.Option)
	}

	libraryValues := C.malloc(C.size_t(len(library_options)) * C.size_t(unsafe.Sizeof(C.uint(0))))
	defer C.free(libraryValues)

	for i, opt := range library_options {
		*(*C.uint)(unsafe.Pointer(uintptr(libraryValues) + uintptr(i)*unsafe.Sizeof(C.uint(0)))) = C.uint(opt.Value)
	}

	return fun(jitOptions, &jitValues, C.uint(len(jit_options)), libraryOptions, &libraryValues, C.uint(len(library_options)))
}

func (lib *CudaLibrary) Unload() error {
	stat := C.cuLibraryUnload(lib.lib)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (lib *CudaLibrary) GetKernel(name string) (*CudaKernel, error) {
	var kernel C.CUkernel
	stat := C.cuLibraryGetKernel(&kernel, lib.lib, C.CString(name))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &CudaKernel{kernel}, nil
}

func (lib *CudaLibrary) GetModule() (*CudaModule, error) {
	var module C.CUmodule
	stat := C.cuLibraryGetModule(&module, lib.lib)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &CudaModule{module}, nil
}

func (lib *CudaLibrary) GetKernels() ([]CudaKernel, error) {
	var count C.uint
	stat := C.cuLibraryGetKernelCount(&count, lib.lib)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	kernels := C.malloc(C.size_t(count) * C.size_t(unsafe.Sizeof(C.CUkernel(nil))))
	defer C.free(kernels)
	stat = C.cuLibraryEnumerateKernels((*C.CUkernel)(kernels), count, lib.lib)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	res := make([]CudaKernel, int(count))
	for i := 0; i < int(count); i++ {
		res[i] = CudaKernel{*(*C.CUkernel)(unsafe.Pointer(uintptr(kernels) + uintptr(i)*unsafe.Sizeof(C.CUkernel(nil))))}
	}
	return res, nil
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