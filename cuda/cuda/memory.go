package cuda

//#include <cuda.h>
import "C"
import (
	"errors"
	"unsafe"
)

type MemAllocation struct {
	Ptr  uintptr
	Size uint64
}

func MemAlloc(size uint64) (*MemAllocation, error) {
	var ptr C.ulonglong
	stat := C.cuMemAlloc(&ptr, C.size_t(size))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &MemAllocation{uintptr(ptr), size}, nil
}

func (ptr *MemAllocation) MemFree() error {
	stat := C.cuMemFree(C.ulonglong(ptr.Ptr))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	ptr.Ptr = 0
	ptr.Size = 0
	return nil
}

func (dev *MemAllocation) MemcpyToDevice(src []byte) error {
	if dev == nil || dev.Ptr == 0 {
		return errors.New("Device memory not allocated")
	}

	if len(src) > int(dev.Size) {
		return errors.New("Source size is greater than device memory size")
	}

	carr := C.CBytes(src) //is a copy needed?
	defer C.free(carr)
	stat := C.cuMemcpyHtoD(C.ulonglong(dev.Ptr), carr, C.size_t(len(src))) //carr = unsafe.Pointer(&src[0])

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (dev *MemAllocation) MemcpyFromDevice(dst []byte) error {

	if dev == nil || dev.Ptr == 0 {
		return errors.New("Device memory not allocated")
	}

	if len(dst) > int(dev.Size) {
		return errors.New("Destination size is greater than device memory size")
	}

	stat := C.cuMemcpyDtoH(unsafe.Pointer(&dst[0]), C.ulonglong(dev.Ptr), C.size_t(len(dst)))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}
