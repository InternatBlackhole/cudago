package cuda

// #include <cuda.h>
import "C"
import "unsafe"

type CoredumpSettings int
type CoredumpGenerationFlags int

func CoredumpGetAttribute(attrib CoredumpSettings) (value []byte, err Result) {
	var size C.size_t
	stat := C.cuCoredumpGetAttribute(C.CUcoredumpSettings(attrib), nil, &size) // get size
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	value = make([]byte, size)
	stat = C.cuCoredumpGetAttribute(C.CUcoredumpSettings(attrib), unsafe.Pointer(&value[0]), &size)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return value, nil
}

func CoredumpGetAttributeGlobal(attrib CoredumpSettings) (value []byte, err Result) {
	var size C.size_t
	stat := C.cuCoredumpGetAttributeGlobal(C.CUcoredumpSettings(attrib), nil, &size) // get size
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	value = make([]byte, size)
	stat = C.cuCoredumpGetAttributeGlobal(C.CUcoredumpSettings(attrib), unsafe.Pointer(&value[0]), &size)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return value, nil
}

func CoredumpSetAttribute(attrib CoredumpSettings, value []byte) Result {
	var size C.size_t = C.size_t(len(value))
	stat := C.cuCoredumpSetAttribute(C.CUcoredumpSettings(attrib), unsafe.Pointer(&value[0]), &size)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func CoredumpSetAttributeGlobal(attrib CoredumpSettings, value []byte) Result {
	var size C.size_t = C.size_t(len(value))
	stat := C.cuCoredumpSetAttributeGlobal(C.CUcoredumpSettings(attrib), unsafe.Pointer(&value[0]), &size)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

const (
	_                               CoredumpSettings = iota
	CU_COREDUMP_ENABLE_ON_EXCEPTION CoredumpSettings = iota
	CU_COREDUMP_TRIGGER_HOST        CoredumpSettings = iota
	CU_COREDUMP_LIGHTWEIGHT         CoredumpSettings = iota
	CU_COREDUMP_ENABLE_USER_TRIGGER CoredumpSettings = iota
	CU_COREDUMP_FILE                CoredumpSettings = iota
	CU_COREDUMP_PIPE                CoredumpSettings = iota
	CU_COREDUMP_GENERATION_FLAGS    CoredumpSettings = iota
	CU_COREDUMP_MAX                 CoredumpSettings = iota
)

const (
	CU_COREDUMP_DEFAULT_FLAGS                CoredumpGenerationFlags = 0
	CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES CoredumpGenerationFlags = (1 << 0)
	CU_COREDUMP_SKIP_GLOBAL_MEMORY           CoredumpGenerationFlags = (1 << 1)
	CU_COREDUMP_SKIP_SHARED_MEMORY           CoredumpGenerationFlags = (1 << 2)
	CU_COREDUMP_SKIP_LOCAL_MEMORY            CoredumpGenerationFlags = (1 << 3)
	CU_COREDUMP_SKIP_ABORT                   CoredumpGenerationFlags = (1 << 4)
	CU_COREDUMP_SKIP_CONSTBANK_MEMORY        CoredumpGenerationFlags = (1 << 5)
	CU_COREDUMP_LIGHTWEIGHT_FLAGS            CoredumpGenerationFlags = CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES |
		CU_COREDUMP_SKIP_GLOBAL_MEMORY |
		CU_COREDUMP_SKIP_SHARED_MEMORY |
		CU_COREDUMP_SKIP_LOCAL_MEMORY |
		CU_COREDUMP_SKIP_CONSTBANK_MEMORY
)
