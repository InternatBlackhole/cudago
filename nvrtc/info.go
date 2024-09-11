package nvrtc

//#include "nvrtc.h"
import "C"
import "unsafe"

// Return the version of the NVRTC library. The version is returned as major and minor version numbers.
func Version() (int, int, error) {
	var major, minor C.int
	err := C.nvrtcVersion(&major, &minor)
	if err != C.NVRTC_SUCCESS {
		return 0, 0, NewNvRtcError(uint32(err))
	}
	return int(major), int(minor), nil
}

func GetSupportedArchs() ([]int, error) {
	var count C.int
	err := C.nvrtcGetNumSupportedArchs(&count)
	if err != C.NVRTC_SUCCESS {
		return nil, NewNvRtcError(uint32(err))
	}
	if count == 0 {
		return nil, nil
	}
	//TODO: try if make([]*C.int, int(count)) works
	archs := C.malloc(C.size_t(count) * C.size_t(C.sizeof_int))
	defer C.free(archs)
	err = C.nvrtcGetSupportedArchs((*C.int)(archs))
	if err != C.NVRTC_SUCCESS {
		return nil, NewNvRtcError(uint32(err))
	}
	archSlice := make([]int, int(count))
	for i := 0; i < int(count); i++ {
		val := (unsafe.Pointer)(uintptr(archs) + uintptr(i)*uintptr(C.sizeof_int))
		archSlice[i] = int(*(*C.int)(val))
	}
	return archSlice, nil
}
