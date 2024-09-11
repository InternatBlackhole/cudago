package nvrtc

//#include "nvrtc.h"
import "C"

type NvRtcError struct {
	code C.nvrtcResult
}

func (e NvRtcError) Error() string {
	var str *C.char = C.nvrtcGetErrorString(e.code)
	return C.GoString(str)
}

func (e NvRtcError) Code() uint32 {
	return uint32(e.code)
}

func NewNvRtcError(code uint32) error {
	return NvRtcError{C.nvrtcResult(code)}
}

func (e NvRtcError) String() string {
	return e.Error()
}
