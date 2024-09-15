package cuda

//perhaps use environ flags in compilation

//#include <cuda.h>
//#include <stdlib.h>
import "C"
import (
	"unsafe"
)

type Dim3 struct {
	X, Y, Z uint32
}

type Kernel struct {
	kern C.CUkernel
}

type FunctionAttribute int

func (k *Kernel) Function() (*Function, Result) {
	var fun C.CUfunction
	stat := C.cuKernelGetFunction(&fun, k.kern)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &Function{fun}, nil
}

func (k *Kernel) GetLibrary() (*Library, Result) {
	var mod C.CUlibrary
	stat := C.cuKernelGetLibrary(&mod, k.kern)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &Library{mod}, nil
}

func (k *Kernel) GetName() (string, Result) {
	var name *C.char = (*C.char)(C.malloc(256))
	defer C.free(unsafe.Pointer(name))
	stat := C.cuKernelGetName((**C.char)(&name), k.kern)

	if stat != C.CUDA_SUCCESS {
		return "", NewCudaError(uint32(stat))
	}
	return C.GoString(name), nil
}

func (k *Kernel) GetParamInfo(paramIndex uint64) (paramOffset uint64, paramSize uint64, err Result) {
	var offset, size C.size_t
	stat := C.cuKernelGetParamInfo(k.kern, C.size_t(paramIndex), &offset, &size)

	if stat != C.CUDA_SUCCESS {
		return 0, 0, NewCudaError(uint32(stat))
	}
	return uint64(offset), uint64(size), nil
}

func (k *Kernel) GetAttribute(attr FunctionAttribute, device *Device) (uint64, Result) {
	var value C.int
	stat := C.cuKernelGetAttribute(&value, C.CUfunction_attribute(attr), k.kern, device.dev)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}
	return uint64(value), nil
}

func (k *Kernel) SetAttribute(attr FunctionAttribute, value int, device *Device) Result {
	stat := C.cuKernelSetAttribute(C.CUfunction_attribute(attr), C.int(value), k.kern, device.dev)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (k *Kernel) SetCacheConfig(config CacheConfig, device *Device) Result {
	stat := C.cuKernelSetCacheConfig(k.kern, C.CUfunc_cache(config), device.dev)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (kernel *Kernel) Launch(grid, block Dim3, args ...unsafe.Pointer) Result {
	return kernel.LaunchEx(grid, block, 0, nil, args...)
}

// TODO: add attributes
func (kernel *Kernel) LaunchEx(grid, block Dim3, sharedMem uint64, stream *Stream /*attributes?,*/, args ...unsafe.Pointer) Result {
	fun, err := kernel.Function()
	if err != nil {
		return err
	}
	return internalLaunchEx(fun.fun, grid, block, sharedMem, stream, args...)
}

func (kernel *Kernel) NativePointer() uintptr {
	return uintptr(unsafe.Pointer(kernel.kern))
}

const (
	// The maximum number of threads per block, beyond which a launch of the function would fail. This number depends on both the function and the device on which the function is currently loaded.
	CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK FunctionAttribute = 0
	// The size in bytes of statically-allocated shared memory required by this function. This does not include dynamically-allocated shared memory requested by the user at runtime.
	CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES FunctionAttribute = 1
	// The size in bytes of user-allocated constant memory required by this function.
	CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES FunctionAttribute = 2
	// The size in bytes of local memory used by each thread of this function.
	CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES FunctionAttribute = 3
	// The number of registers used by each thread of this function.
	CU_FUNC_ATTRIBUTE_NUM_REGS FunctionAttribute = 4
	// The PTX virtual architecture version for which the function was compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA 3.0.
	CU_FUNC_ATTRIBUTE_PTX_VERSION FunctionAttribute = 5
	// The binary architecture version for which the function was compiled. This value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary architecture version.
	CU_FUNC_ATTRIBUTE_BINARY_VERSION FunctionAttribute = 6
	// The attribute to indicate whether the function has been compiled with user specified option "-Xptxas --dlcmFunctionAttribute =ca" set .
	CU_FUNC_ATTRIBUTE_CACHE_MODE_CA FunctionAttribute = 7
	// The maximum size in bytes of dynamically-allocated shared memory that can be used by this function. If the user-specified dynamic shared memory size is larger than this value, the launch will fail. See cuFuncSetAttribute, cuKernelSetAttribute
	CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES FunctionAttribute = 8
	// On devices where the L1 cache and shared memory use the same hardware resources, this sets the shared memory carveout preference, in percent of the total shared memory. Refer to CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR. This is only a hint, and the driver can choose a different ratio if required to execute the function. See cuFuncSetAttribute, cuKernelSetAttribute
	CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT FunctionAttribute = 9
	// If this attribute is set, the kernel must launch with a valid cluster size specified. See cuFuncSetAttribute, cuKernelSetAttribute
	CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET FunctionAttribute = 10
	// The required cluster width in blocks. The values must either all be 0 or all be positive. The validity of the cluster dimensions is otherwise checked at launch time.If the value is set during compile time, it cannot be set at runtime. Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED. See cuFuncSetAttribute, cuKernelSetAttribute
	CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH FunctionAttribute = 11
	// The required cluster height in blocks. The values must either all be 0 or all be positive. The validity of the cluster dimensions is otherwise checked at launch time.If the value is set during compile time, it cannot be set at runtime. Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED. See cuFuncSetAttribute, cuKernelSetAttribute
	CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT FunctionAttribute = 12
	// The required cluster depth in blocks. The values must either all be 0 or all be positive. The validity of the cluster dimensions is otherwise checked at launch time.If the value is set during compile time, it cannot be set at runtime. Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED. See cuFuncSetAttribute, cuKernelSetAttribute
	CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH FunctionAttribute = 13
	// Whether the function can be launched with non-portable cluster size. 1 is allowed, 0 is disallowed. A non-portable cluster size may only function on the specific SKUs the program is tested on. The launch might fail if the program is run on a different hardware platform.CUDA API provides cudaOccupancyMaxActiveClusters to assist with checking whether the desired size can be launched on the current device.Portable Cluster SizeA portable cluster size is guaranteed to be functional on all compute capabilities higher than the target compute capability. The portable cluster size for sm_90 is 8 blocks per cluster. This value may increase for future compute capabilities.The specific hardware unit may support higher cluster sizes that’s not guaranteed to be portable. See cuFuncSetAttribute, cuKernelSetAttribute
	CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED FunctionAttribute = 14
	// The block scheduling policy of a function. The value type is CUclusterSchedulingPolicy / cudaClusterSchedulingPolicy. See cuFuncSetAttribute, cuKernelSetAttribute
	CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE FunctionAttribute = 15
	// The maximum value for CUfunction_attribute.
	CU_FUNC_ATTRIBUTE_MAX = 16
)
