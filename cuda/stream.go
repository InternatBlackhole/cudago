package cuda

//#include <cuda.h>
import "C"
import (
	"unsafe"
)

type Stream struct {
	stream C.CUstream
}

type StreamCallback func(*Stream, uint32, *any)
type StreamWaitFlag int
type StreamFlags int
type StreamCaptureMode int

func CreateStream(flags StreamFlags) (*Stream, Result) {
	var stream C.CUstream
	stat := C.cuStreamCreate(&stream, C.uint(flags))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &Stream{stream}, nil
}

func CreateStreamWithPriority(flags StreamFlags, priority int) (*Stream, Result) {
	var stream C.CUstream
	stat := C.cuStreamCreateWithPriority(&stream, C.uint(flags), C.int(priority))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &Stream{stream}, nil
}

func (s *Stream) NativePointer() uintptr {
	return uintptr(unsafe.Pointer(s.stream))
}

func (s *Stream) Destroy() Result {
	stat := C.cuStreamDestroy(s.stream)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (s *Stream) CopyAttributes(dst *Stream) Result {
	stat := C.cuStreamCopyAttributes(s.stream, dst.stream)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (s *Stream) AddCallback(callback StreamCallback, userData *any, flags StreamFlags) Result {
	return ErrDeprecated
}

// TODO: Implement this when memory is implemented
func (s *Stream) AttachMemAsync(dptr uintptr, length uint64, flags StreamFlags) Result {
	return ErrUnsupported
	stat := C.cuStreamAttachMemAsync(s.stream, C.CUdeviceptr(dptr), C.size_t(length), C.uint(flags))
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (s *Stream) Synchronize() Result {
	stat := C.cuStreamSynchronize(s.stream)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (s *Stream) Query() (completed bool, err Result) {
	stat := C.cuStreamQuery(s.stream)
	if stat == C.CUDA_SUCCESS || stat == C.CUDA_ERROR_NOT_READY {
		return stat == C.CUDA_SUCCESS, nil
	}
	return false, NewCudaError(uint32(stat))
}

func (s *Stream) WaitEvent(event *Event, flags StreamWaitFlag) Result {
	stat := C.cuStreamWaitEvent(s.stream, event.event, C.uint(flags))
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (s *Stream) GetPriority() (int, Result) {
	var priority C.int
	stat := C.cuStreamGetPriority(s.stream, &priority)
	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}
	return int(priority), nil
}

func (s *Stream) GetId() (uint64, Result) {
	var id C.ulonglong
	stat := C.cuStreamGetId(s.stream, &id)
	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}
	return uint64(id), nil
}

func (s *Stream) GetContext() (*Context, Result) {
	var ctx C.CUcontext
	stat := C.cuStreamGetCtx(s.stream, &ctx)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &Context{ctx}, nil
}

func (s *Stream) GetFlags() (StreamFlags, Result) {
	var flags C.uint
	stat := C.cuStreamGetFlags(s.stream, &flags)
	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}
	return StreamFlags(flags), nil
}

/*TODO: Implement this if you implement graphs
func (s *Stream) BeginCapture(mode StreamCaptureMode) Result {
	stat := C.cuStreamBeginCapture(s.stream, C.CUstreamCaptureMode(mode))
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (s *Stream) EndCapture() Result {
	stat := C.cuStreamEndCapture(s.stream)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}*/

const (
	CU_STREAM_CAPTURE_MODE_GLOBAL       StreamCaptureMode = 0
	CU_STREAM_CAPTURE_MODE_THREAD_LOCAL StreamCaptureMode = 1
	CU_STREAM_CAPTURE_MODE_RELAXED      StreamCaptureMode = 2
)

const (
	CU_STREAM_DEFAULT      StreamFlags = 0x0 // Default stream flag
	CU_STREAM_NON_BLOCKING StreamFlags = 0x1 // Stream does not synchronize with stream 0 (the NULL stream)
)

const (
	CU_EVENT_WAIT_DEFAULT  StreamWaitFlag = 0x0 // Default event wait flag
	CU_EVENT_WAIT_EXTERNAL StreamWaitFlag = 0x1 // When using stream capture, create an event wait node instead of the default behavior. This flag is invalid when used outside of capture.
)
