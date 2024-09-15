package cuda

//#include <cuda.h>
import "C"

type MulticastMemory struct {
	h C.CUmemGenericAllocationHandle
}

type MulticastGranularityFlags int
type FlagAllocationHandleType int

func CreateMulticastMemory(memSize uint64, numDevices uint32, handles FlagAllocationHandleType) (*MulticastMemory, Result) {
	var h C.CUmemGenericAllocationHandle
	prop := C.CUmulticastObjectProp{
		numDevices:  C.uint(numDevices),
		handleTypes: C.ulonglong(handles),
		flags:       0, // as per the documentation, this should be 0
		size:        C.size_t(memSize),
	}
	stat := C.cuMulticastCreate(&h, &prop)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &MulticastMemory{h}, nil
}

func (mem *MulticastMemory) AddDevice(device *Device) Result {
	stat := C.cuMulticastAddDevice(mem.h, device.dev)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (mem *MulticastMemory) Unbind(device *Device, mcOffset, size uint64) Result {
	stat := C.cuMulticastUnbind(mem.h, device.dev, C.size_t(mcOffset), C.size_t(size))
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (mem *MulticastMemory) BindAddr() {
	//TODO: Implement this
}

func (mem *MulticastMemory) BindMem() {
	//TODO: Implement this
}

func (mem *MulticastMemory) NativePointer() uintptr {
	return uintptr(mem.h)
}

func MulticastGetGranularity(memSize uint64, numDevices uint32,
	handles FlagAllocationHandleType, options MulticastGranularityFlags) (granularity uint64, err Result) {
	var gran C.size_t
	prop := C.CUmulticastObjectProp{
		numDevices:  C.uint(numDevices),
		handleTypes: C.ulonglong(handles),
		flags:       0, // as per the documentation, this should be 0
		size:        C.size_t(memSize),
	}
	stat := C.cuMulticastGetGranularity(&gran, &prop, C.CUmulticastGranularity_flags(options))
	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}
	return uint64(gran), nil
}

const (
	//Does not allow any export mechanism. >
	CU_MEM_HANDLE_TYPE_NONE FlagAllocationHandleType = 0x0
	//Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int)
	CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR FlagAllocationHandleType = 0x1
	// Allows a Win32 NT handle to be used for exporting. (HANDLE)
	CU_MEM_HANDLE_TYPE_WIN32 FlagAllocationHandleType = 0x2
	// Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)
	CU_MEM_HANDLE_TYPE_WIN32_KMT FlagAllocationHandleType = 0x4
	// Allows a fabric handle to be used for exporting. (CUmemFabricHandle)
	CU_MEM_HANDLE_TYPE_FABRIC FlagAllocationHandleType = 0x8

	CU_MEM_HANDLE_TYPE_MAX FlagAllocationHandleType = 0x7FFFFFFF
)

const (
	// Minimum required granularity
	CU_MULTICAST_GRANULARITY_MINIMUM MulticastGranularityFlags = 0x0
	// Recommended granularity for best performance
	CU_MULTICAST_GRANULARITY_RECOMMENDED MulticastGranularityFlags = 0x1
)
