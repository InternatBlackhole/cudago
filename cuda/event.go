package cuda

//#include <cuda.h>
//#include <stdlib.h>
import "C"
import "unsafe"

// Represents a CUDA event
type Event struct {
	event C.CUevent
}

// Event flags
type EventFlag int

// Event record flags
type EventRecordFlag int

// Creates an event with the default flag.
func NewEvent() (*Event, Result) {
	return NewEventCustomFlags(CU_EVENT_DEFAULT)
}

// Creates an event with custom flags.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db
func NewEventCustomFlags(event_flags EventFlag) (*Event, Result) {
	var event C.CUevent
	stat := C.cuEventCreate(&event, C.uint(event_flags))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Event{event}, nil
}

// Destroys an event.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef
func (e *Event) Destroy() Result {
	stat := C.cuEventDestroy(e.event)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Records an event.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1
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

// Records an event with flags.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b
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

// Waits for an event to complete, all work preceding the event in the current stream is guaranteed to complete before the event is completed.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c
func (e *Event) Synchronize() Result {
	stat := C.cuEventSynchronize(e.event)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Queries an event's status.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef
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

// Returns the native pointer of the event.
func (e *Event) NativePointer() uintptr {
	return uintptr(unsafe.Pointer(e.event))
}

// Computes the elapsed time between two events in milliseconds with a resolution of approximately 0.5 microseconds.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97
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
