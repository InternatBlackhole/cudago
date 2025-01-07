package cuda

//#include <cuda.h>
import "C"
import (
	"unsafe"
)

// All possible number types in Go.
// For use in memory allocation functions.
type Number interface {
	int | int8 | int16 | int32 | int64 |
		uint | uint8 | uint16 | uint32 | uint64 | uintptr |
		float32 | float64
}

// Specifies that this object can be freed.
type Freeable interface {
	Free() Result
}

// Represents a memory allocation that can be copied to and from the device.
type Memory interface {
	Freeable
	MemcpyFromDevice(dst []byte) Result
	MemcpyToDevice(src []byte) Result
	//Memset(value byte) Result
}

// Represents a memory allocation on the device.
type DeviceMemory struct {
	Ptr      uintptr //CUdeviceptr
	Size     uint64
	freeable bool
}

// Represents a memory allocation on the host.
type HostMemory[T Number] struct {
	Ptr        uintptr //address of the first byte
	Arr        []T
	ActualSize uint64 // number of bytes allocated for Arr
	registered bool
}

// Represents a managed/unified memory allocation.
type ManagedMemory[T Number] struct {
	Ptr        uintptr
	Arr        []T
	ActualSize uint64
}

// Flags for memory allocation.
type MemAttachFlag int

// Flags for host memory allocation.
type HostMemAllocFlag int

// Flags for host memory registration.
type HostMemRegisterFlag int

// Wraps a device memory pointer to this library's DeviceMemory type.
func WrapAllocationDevice(ptr uintptr, size uint64, freeable bool) *DeviceMemory {
	return &DeviceMemory{ptr, size, freeable}
}

// Registers your host memory with CUDA.
// It is your responsibility to free the memory when you are done with it (after unregistering it).
func RegisterAllocationHost[T Number](ptr []T, elemSize uint64, flags HostMemRegisterFlag) (*HostMemory[T], Result) {
	len := uint64(len(ptr))
	actualSize := len * elemSize
	firstElemAddr := unsafe.Pointer(&ptr[0])
	stat := C.cuMemHostRegister(firstElemAddr, C.size_t(actualSize), C.uint(flags))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &HostMemory[T]{uintptr(firstElemAddr), unsafe.Slice((*T)(firstElemAddr), len), actualSize, true}, nil
}

// Allocates size bytes of device memory.
func DeviceMemAlloc(size uint64) (*DeviceMemory, Result) {
	var ptr C.ulonglong
	stat := C.cuMemAlloc(&ptr, C.size_t(size))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &DeviceMemory{uintptr(ptr), size, true}, nil
}

// Frees memory allocated on the device.
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

// Copies srcSize bytes of data from the host located at src to the device memory allocation.
func (dev *DeviceMemory) MemcpyToDevice(src unsafe.Pointer, srcSize uint64) Result {
	if dev == nil || dev.Ptr == 0 {
		return newInternalError("invalid device memory")
	}

	if srcSize > dev.Size {
		return newInternalError("source size is greater than device memory size")
	}

	stat := C.cuMemcpyHtoD(C.ulonglong(dev.Ptr), src, C.size_t(srcSize))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Copies dstSize bytes of data from the device memory allocation to the host located at dst.
func (dev *DeviceMemory) MemcpyFromDevice(dst unsafe.Pointer, dstSize uint64) Result {

	if dev == nil || dev.Ptr == 0 {
		return newInternalError("invalid device memory")
	}

	if dstSize > dev.Size {
		return newInternalError("destination size is greater than device memory size")
	}

	stat := C.cuMemcpyDtoH(dst, C.ulonglong(dev.Ptr), C.size_t(dstSize))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Allocates size bytes of host memory with flags.
func HostMemAllocWithFlags[T Number](len uint64, elemSize uint64, flags HostMemAllocFlag) (*HostMemory[T], Result) {
	var ptr unsafe.Pointer
	size := len * elemSize
	stat := C.cuMemHostAlloc(&ptr, C.size_t(size), C.uint(flags))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &HostMemory[T]{uintptr(ptr), unsafe.Slice((*T)(ptr), len), size, false}, nil
}

// Allocates size bytes of host memory.
// This method of allocation page-lock the memory thus allowing for faster transfers between the host and device.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0
func HostMemAlloc[T Number](len uint64, elemSize uint64) (*HostMemory[T], Result) {
	var ptr unsafe.Pointer
	size := len * elemSize
	stat := C.cuMemAllocHost(&ptr, C.size_t(size))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &HostMemory[T]{uintptr(ptr), unsafe.Slice((*T)(ptr), len), size, false}, nil
}

// Frees memory allocated on the host.
func (ptr *HostMemory[T]) Free() Result {
	if ptr.Arr == nil || ptr.ActualSize == 0 {
		return nil
	}

	if ptr.registered {
		stat := C.cuMemHostUnregister(unsafe.Pointer(&ptr.Arr[0]))
		if stat != C.CUDA_SUCCESS {
			return NewCudaError(uint32(stat))
		}
	} else {
		stat := C.cuMemFreeHost(unsafe.Pointer(&ptr.Arr[0]))

		if stat != C.CUDA_SUCCESS {
			return NewCudaError(uint32(stat))
		}
	}

	ptr.Ptr = 0
	ptr.Arr = nil
	ptr.ActualSize = 0
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

// Returns the host memory allocation as a byte slice.
func (ptr *HostMemory[T]) AsByteSlice() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&ptr.Arr[0])), int(ptr.ActualSize))
}

// Allocates size bytes of managed memory with flags.
func ManagedMemAllocFlags[T Number](elems uint64, elemSize uint64, flags MemAttachFlag) (*ManagedMemory[T], Result) {
	var ptr C.CUdeviceptr
	size := elems * elemSize
	stat := C.cuMemAllocManaged(&ptr, C.size_t(size), C.uint(flags))
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &ManagedMemory[T]{uintptr(ptr), unsafe.Slice((*T)(unsafe.Pointer(uintptr(ptr))), elems), size}, nil
}

// Allocates size bytes of managed memory.
func ManagedMemAlloc[T Number](elems uint64, elemSize uint64) (*ManagedMemory[T], Result) {
	return ManagedMemAllocFlags[T](elems, elemSize, CU_MEM_ATTACH_GLOBAL)
}

// Frees allocated managed memory.
func (ptr *ManagedMemory[T]) Free() Result {
	if ptr == nil || ptr.Ptr == 0 {
		return nil
	}

	stat := C.cuMemFree(C.CUdeviceptr(ptr.Ptr))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	ptr.Ptr = 0
	ptr.ActualSize = 0
	ptr.Arr = nil
	return nil
}

// Returns the managed memory allocation as a byte slice.
func (ptr *ManagedMemory[T]) AsByteSlice() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(ptr.Ptr)), int(ptr.ActualSize))
}

// Copies size bytes of data from src to dst.
// Can be used to copy data between device and host memory or between device memory allocations.
func MemCpy(dst uintptr, src uintptr, size uint64) Result {
	stat := C.cuMemcpy(C.ulonglong(dst), C.ulonglong(src), C.size_t(size))
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

// Copies size bytes of data from src to dst asynchronously.
// Can be used to copy data between device and host memory or between device memory allocations.
func MemCpyAsync(dst uintptr, src uintptr, size uint64, stream *Stream) Result {
	str := C.CUstream(nil)
	if stream != nil {
		str = stream.stream
	}
	stat := C.cuMemcpyAsync(C.ulonglong(dst), C.ulonglong(src), C.size_t(size), str)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
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
