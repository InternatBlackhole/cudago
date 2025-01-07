package cuda

//#include <cuda.h>
import "C"
import "unsafe"

// Represents a CUDA linker state
type LinkState struct {
	state   C.CUlinkState
	options []C.uint // have to stay alive
}

// JIT Option
type JitInputType int

// Creates a pending JIT linker invocation.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g86ca4052a2fab369cb943523908aa80d
func NewCudaLinkState(linkOptions []JitOption) (*LinkState, Result) {
	var state C.CUlinkState
	_, vals, optsAddr, valsAddr := parseJitOptions(linkOptions)

	stat := C.cuLinkCreate(C.uint(len(linkOptions)), optsAddr, valsAddr, &state)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &LinkState{state, vals}, nil
}

// Destroys state for a JIT linker invocation.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g01b7ae2a34047b05716969af245ce2d9
func (l *LinkState) Destroy() Result {
	stat := C.cuLinkDestroy(l.state)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

// Complete a pending linker invocation.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g818fcd84a4150a997c0bba76fef4e716
func (l *LinkState) Complete() (cubin []byte, err Result) {
	var _cubin unsafe.Pointer = nil
	var cubinSize C.size_t
	stat := C.cuLinkComplete(l.state, &_cubin, &cubinSize)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return C.GoBytes(_cubin, C.int(cubinSize)), nil
}

// Add an input to a pending linker invocation.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g3ebcd2ccb772ba9c120937a2d2831b77
func (l *LinkState) AddData(data []byte, kind JitInputType, name string, options []JitOption) Result {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	_, _, optsAddr, valsAddr := parseJitOptions(options)

	stat := C.cuLinkAddData(l.state, C.CUjitInputType(kind), unsafe.Pointer(&data[0]),
		C.size_t(len(data)), cName, C.uint(len(options)), optsAddr, valsAddr)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

// Add a file input to a pending linker invocation.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g1224c0fd48d4a683f3ce19997f200a8c
func (l *LinkState) AddFile(path string, kind JitInputType, options []JitOption) Result {
	cfilename := C.CString(path)
	defer C.free(unsafe.Pointer(cfilename))

	_, _, optsAddr, valsAddr := parseJitOptions(options)

	stat := C.cuLinkAddFile(l.state, C.CUjitInputType(kind), cfilename,
		C.uint(len(options)), optsAddr, valsAddr)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

// Returns the native handle of the CUDA linker state.
func (l *LinkState) NativePointer() uintptr {
	return uintptr(unsafe.Pointer(l.state))
}

const (
	CU_JIT_INPUT_CUBIN     JitInputType = 0 // Compiled device-class-specific device code Applicable options: none
	CU_JIT_INPUT_PTX       JitInputType = 1 // PTX source code Applicable options: PTX compiler options
	CU_JIT_INPUT_FATBINARY JitInputType = 2 // Bundle of multiple cubins and/or PTX of some device code Applicable options: PTX compiler options, CU_JIT_FALLBACK_STRATEGY
	CU_JIT_INPUT_OBJECT    JitInputType = 3 // Host object with embedded device code Applicable options: PTX compiler options, CU_JIT_FALLBACK_STRATEGY
	CU_JIT_INPUT_LIBRARY   JitInputType = 4 // Archive of host objects with embedded device code Applicable options: PTX compiler options, CU_JIT_FALLBACK_STRATEGY
	CU_JIT_INPUT_NVVM      JitInputType = 5 // Deprecated. High-level intermediate code for link-time optimization Applicable options: NVVM compiler options, PTX compiler options. Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
	CU_JIT_NUM_INPUT_TYPES JitInputType = 6
)
