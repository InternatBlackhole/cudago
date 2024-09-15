package cuda

//#include <cuda.h>
import "C"
import (
	"unsafe"
)

type Freeable interface {
	Free() Result
}

type Memory interface {
	Freeable
	MemcpyFromDevice(dst []byte) Result
	MemcpyToDevice(src []byte) Result
	//Memset(value byte) Result
}

type DeviceMemory struct {
	Ptr      uintptr //CUdeviceptr
	Size     uint64
	freeable bool
}

type HostMemory struct {
	Ptr        uintptr
	Size       uint64
	registered bool
}

// TODO: expose this as and []byte or []int8 so that user can copy into it
type ManagedMemory struct {
	Ptr  uintptr
	Size uint64
}

type MemAttachFlag int
type HostMemAllocFlag int
type HostMemRegisterFlag int

func WrapAllocationDevice(ptr uintptr, size uint64, freeable bool) *DeviceMemory {
	return &DeviceMemory{ptr, size, freeable}
}

// Registers your host memory with CUDA. It is your responsibility to free the memory when you are done with it (after unregistering it).
func RegisterAllocationHost(ptr *any, size uint64, flags HostMemRegisterFlag) (*HostMemory, Result) {
	stat := C.cuMemHostRegister(unsafe.Pointer(ptr), C.size_t(size), C.uint(flags))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &HostMemory{uintptr(unsafe.Pointer(ptr)), uint64(size), true}, nil
}

func WrapAllocationManaged(ptr uintptr, size uint64) *ManagedMemory {
	return &ManagedMemory{ptr, size}
}

func DeviceMemAlloc(size uint64) (*DeviceMemory, Result) {
	var ptr C.ulonglong
	stat := C.cuMemAlloc(&ptr, C.size_t(size))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &DeviceMemory{uintptr(ptr), size, true}, nil
}

func (ptr *DeviceMemory) Free() Result {
	if ptr == nil || ptr.Ptr == 0 {
		return nil
	}

	if !ptr.freeable {
		return nil
	}

	stat := C.cuMemFree(C.ulonglong(ptr.Ptr))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	ptr.Ptr = 0
	ptr.Size = 0
	return nil
}

func (dev *DeviceMemory) MemcpyToDevice(src []byte) Result {
	if dev == nil || dev.Ptr == 0 {
		return newInternalError("invalid device memory")
	}

	if len(src) > int(dev.Size) {
		return newInternalError("source size is greater than device memory size")
	}

	stat := C.cuMemcpyHtoD(C.ulonglong(dev.Ptr), unsafe.Pointer(&src[0]), C.size_t(len(src)))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (dev *DeviceMemory) MemcpyFromDevice(dst []byte) Result {

	if dev == nil || dev.Ptr == 0 {
		return newInternalError("invalid device memory")
	}

	if len(dst) > int(dev.Size) {
		return newInternalError("destination size is greater than device memory size")
	}

	stat := C.cuMemcpyDtoH(unsafe.Pointer(&dst[0]), C.ulonglong(dev.Ptr), C.size_t(len(dst)))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func HostMemAllocWithFlags(size uint64, flags HostMemAllocFlag) (*HostMemory, Result) {
	var ptr unsafe.Pointer
	stat := C.cuMemHostAlloc(&ptr, C.size_t(size), C.uint(flags))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &HostMemory{uintptr(ptr), size, true}, nil
}

func HostMemAlloc(size uint64) (*HostMemory, Result) {
	var ptr unsafe.Pointer
	stat := C.cuMemAllocHost(&ptr, C.size_t(size))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &HostMemory{uintptr(ptr), size, false}, nil
}

func (ptr *HostMemory) Free() Result {
	if ptr == nil || ptr.Ptr == 0 {
		return nil
	}

	if ptr.registered {
		stat := C.cuMemHostUnregister(unsafe.Pointer(ptr.Ptr))
		if stat != C.CUDA_SUCCESS {
			return NewCudaError(uint32(stat))
		}
	} else {
		stat := C.cuMemFreeHost(unsafe.Pointer(ptr.Ptr))

		if stat != C.CUDA_SUCCESS {
			return NewCudaError(uint32(stat))
		}
	}

	ptr.Ptr = 0
	ptr.Size = 0
	return nil
}

/*func (ptr *HostMemory) MemcpyToDevice(src []byte) Result {
	if ptr == nil || ptr.Ptr == 0 {
		return newInternalError("invalid host memory")
	}

	if len(src) > int(ptr.Size) {
		return newInternalError("source size is greater than host memory size")
	}

	stat := C.cuMemcpyHtoD(C.ulonglong(ptr.Ptr), unsafe.Pointer(&src[0]), C.size_t(len(src)))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (ptr *HostMemory) MemcpyFromDevice(dst []byte) Result {
	if ptr == nil || ptr.Ptr == 0 {
		return newInternalError("invalid host memory")
	}

	if len(dst) > int(ptr.Size) {
		return newInternalError("destination size is greater than host memory size")
	}

	stat := C.cuMemcpyDtoH(unsafe.Pointer(&dst[0]), C.ulonglong(ptr.Ptr), C.size_t(len(dst)))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}*/

func (ptr *HostMemory) AsSlice() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(ptr.Ptr)), int(ptr.Size))
}

func ManagedMemAlloc(size uint64, flags MemAttachFlag) (*ManagedMemory, Result) {
	var ptr C.CUdeviceptr
	stat := C.cuMemAllocManaged(&ptr, C.size_t(size), C.uint(flags))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &ManagedMemory{uintptr(ptr), size}, nil
}

func (ptr *ManagedMemory) Free() Result {
	if ptr == nil || ptr.Ptr == 0 {
		return nil
	}

	stat := C.cuMemFree(C.CUdeviceptr(ptr.Ptr))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	ptr.Ptr = 0
	ptr.Size = 0
	return nil
}

func (ptr *ManagedMemory) AsSlice() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(ptr.Ptr)), int(ptr.Size))
}

const (
	CU_MEM_ATTACH_GLOBAL MemAttachFlag = 0x1 // Memory can be accessed by any stream on any device
	CU_MEM_ATTACH_HOST   MemAttachFlag = 0x2 // Memory cannot be accessed by any stream on any device
	CU_MEM_ATTACH_SINGLE MemAttachFlag = 0x4 // Memory can only be accessed by a single stream on the associated device
)

const (
	// If set, host memory is portable between CUDA contexts.
	CU_MEMHOSTALLOC_PORTABLE HostMemAllocFlag = 0x01
	// If set, host memory is mapped into CUDA address space and cuMemHostGetDevicePointer() may be called on the host pointer.
	CU_MEMHOSTALLOC_DEVICEMAP HostMemAllocFlag = 0x02
	// If set, host memory is allocated as write-combined - fast to write, faster to DMA, slow to read except via SSE4 streaming load instruction (MOVNTDQA).
	CU_MEMHOSTALLOC_WRITECOMBINED HostMemAllocFlag = 0x04
)

const (
	// If set, host memory is portable between CUDA contexts.
	CU_MEMHOSTREGISTER_PORTABLE HostMemRegisterFlag = 0x01
	// If set, host memory is mapped into CUDA address space and cuMemHostGetDevicePointer() may be called on the host pointer.
	CU_MEMHOSTREGISTER_DEVICEMAP HostMemRegisterFlag = 0x02
	// If set, the passed memory pointer is treated as pointing to some memory-mapped I/O space, e.g. belonging to a third-party PCIe device. On Windows the flag is a no-op. On Linux that memory is marked as non cache-coherent for the GPU and is expected to be physically contiguous. It may return CUDA_ERROR_NOT_PERMITTED if run as an unprivileged user, CUDA_ERROR_NOT_SUPPORTED on older Linux kernel versions. On all other platforms, it is not supported and CUDA_ERROR_NOT_SUPPORTED is returned.
	CU_MEMHOSTREGISTER_IOMEMORY HostMemRegisterFlag = 0x04
	// If set, the passed memory pointer is treated as pointing to memory that is considered read-only by the device. On platforms without CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, this flag is required in order to register memory mapped to the CPU as read-only. Support for the use of this flag can be queried from the device attribute CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED. Using this flag with a current context associated with a device that does not have this attribute set will cause cuMemHostRegister to error with CUDA_ERROR_NOT_SUPPORTED.
	CU_MEMHOSTREGISTER_READ_ONLY HostMemRegisterFlag = 0x08
)
