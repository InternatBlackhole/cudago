package cuda

//#include <cuda.h>
import "C"
import "unsafe"

type LinkState struct {
	state   C.CUlinkState
	options []C.uint // have to stay alive
}

type JitInputType int

func NewCudaLinkState(linkOptions []JitOption) (*LinkState, Result) {
	var state C.CUlinkState
	_, vals, optsAddr, valsAddr := parseJitOptions(linkOptions)

	stat := C.cuLinkCreate(C.uint(len(linkOptions)), optsAddr, valsAddr, &state)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &LinkState{state, vals}, nil
}

func (l *LinkState) Destroy() Result {
	stat := C.cuLinkDestroy(l.state)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (l *LinkState) Complete() (cubin []byte, err Result) {
	var _cubin unsafe.Pointer = nil
	var cubinSize C.size_t
	stat := C.cuLinkComplete(l.state, &_cubin, &cubinSize)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return C.GoBytes(_cubin, C.int(cubinSize)), nil
}

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
