package cuda

//#include <cuda.h>
import "C"
import "unsafe"

type CudaDevice struct {
	dev C.CUdevice
}

func DeviceCount() (int, error) {
	var count C.int
	err := C.cuDeviceGetCount(&count)
	if err != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(err))
	}
	return int(count), nil
}

func DeviceGet(device int) (*CudaDevice, error) {
	var dev C.CUdevice
	err := C.cuDeviceGet(&dev, C.int(device))
	if err != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(err))
	}
	return &CudaDevice{dev}, nil
}

func (dev *CudaDevice) Name() (string, error) {
	defaultSize := 256
	name := (*C.char)(C.malloc(C.ulong(defaultSize)))
	defer C.free(unsafe.Pointer(name))
	err := C.cuDeviceGetName(name, C.int(defaultSize), dev.dev)
	if err != C.CUDA_SUCCESS {
		return "", NewCudaError(uint32(err))
	}
	return C.GoString(name), nil
}

func (dev *CudaDevice) TotalMem() (uint64, error) {
	var mem C.size_t
	err := C.cuDeviceTotalMem(&mem, dev.dev)
	if err != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(err))
	}
	return uint64(mem), nil
}

func (dev *CudaDevice) UUID() (string, error) {
	uuid := make([]byte, 16)
	err := C.cuDeviceGetUuid((*C.CUuuid)(unsafe.Pointer(&uuid[0])), dev.dev)
	if err != C.CUDA_SUCCESS {
		return "", NewCudaError(uint32(err))
	}
	return string(uuid), nil
}
