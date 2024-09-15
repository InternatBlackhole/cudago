package cuda

//#include <cuda.h>
//#include <stdlib.h>
import "C"
import "unsafe"

type Event struct {
	event C.CUevent
}

type EventFlag int
type EventRecordFlag int

func NewEvent() (*Event, Result) {
	return NewEventCustomFlags(CU_EVENT_DEFAULT)
}

func NewEventCustomFlags(event_flags EventFlag) (*Event, Result) {
	var event C.CUevent
	stat := C.cuEventCreate(&event, C.uint(event_flags))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Event{event}, nil
}

func (e *Event) Destroy() Result {
	stat := C.cuEventDestroy(e.event)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (e *Event) Record(stream *Stream) Result {
	var str C.CUstream = nil
	if stream != nil {
		str = stream.stream
	}
	stat := C.cuEventRecord(e.event, str)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (e *Event) RecordWithFlags(stream *Stream, flags EventRecordFlag) Result {
	var str C.CUstream = nil
	if stream != nil {
		str = stream.stream
	}
	stat := C.cuEventRecordWithFlags(e.event, str, C.uint(flags))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (e *Event) Synchronize() Result {
	stat := C.cuEventSynchronize(e.event)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (e *Event) Query() (completed bool, err Result) {
	stat := C.cuEventQuery(e.event)

	if stat == C.CUDA_SUCCESS {
		return true, nil
	} else if stat == C.CUDA_ERROR_NOT_READY {
		return false, nil
	} else {
		return false, NewCudaError(uint32(stat))
	}
}

func (e *Event) NativePointer() uintptr {
	return uintptr(unsafe.Pointer(e.event))
}

func EventElapsedTime(start, end *Event) (float32, Result) {
	var time float32
	stat := C.cuEventElapsedTime((*C.float)(&time), start.event, end.event)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return time, nil
}

const (
	CU_EVENT_DEFAULT        EventFlag = 0x0 // Default event flag
	CU_EVENT_BLOCKING_SYNC  EventFlag = 0x1 // Event uses blocking synchronization
	CU_EVENT_DISABLE_TIMING EventFlag = 0x2 // Event will not record timing data
	CU_EVENT_INTERPROCESS   EventFlag = 0x4 // Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set
)

const (
	CU_EVENT_RECORD_DEFAULT  EventRecordFlag = 0x0 // Default event record flag
	CU_EVENT_RECORD_EXTERNAL EventRecordFlag = 0x1 //When using stream capture, create an event record node instead of the default behavior. This flag is invalid when used outside of capture.
)
