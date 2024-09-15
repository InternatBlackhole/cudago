package cuda

//#include <cuda.h>
import "C"
import "unsafe"

type Function struct {
	fun C.CUfunction
}

type LauncAttributeID int

type LaunchAttribute struct {
	ID    LauncAttributeID
	Value unsafe.Pointer
}

/*func (f *Function) AsKernel() (*Kernel, Result) {
	return &Kernel{C.CUkernel(unsafe.Pointer(f.fun))}, nil
}*/

func (f *Function) Load() Result {
	return NewCudaError(uint32(C.cuFuncLoad(f.fun)))
}

func (f *Function) IsLoaded() (bool, Result) {
	var loaded C.CUfunctionLoadingState
	stat := C.cuFuncIsLoaded(&loaded, f.fun)
	if stat != C.CUDA_SUCCESS {
		return false, NewCudaError(uint32(stat))
	}

	return loaded != 0, nil
}

func (f *Function) GetName() (string, Result) {
	var name *C.char
	stat := C.cuFuncGetName(&name, f.fun)
	if stat != C.CUDA_SUCCESS {
		return "", NewCudaError(uint32(stat))
	}

	return C.GoString(name), nil
}

func (f *Function) GetModule() (*Module, Result) {
	var mod C.CUmodule
	stat := C.cuFuncGetModule(&mod, f.fun)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Module{mod}, nil
}

func (f *Function) GetAttribute(attr FunctionAttribute) (int, Result) {
	var value C.int
	stat := C.cuFuncGetAttribute(&value, C.CUfunction_attribute(attr), f.fun)
	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return int(value), nil
}

func (f *Function) GetParamInfo(index uint64) (offset uint64, size uint64, err Result) {
	var off, siz C.size_t
	stat := C.cuFuncGetParamInfo(f.fun, C.size_t(index), &off, &siz)
	if stat != C.CUDA_SUCCESS {
		return 0, 0, NewCudaError(uint32(stat))
	}

	return uint64(off), uint64(siz), nil
}

func (f *Function) SetCacheConfig(config CacheConfig) Result {
	stat := C.cuFuncSetCacheConfig(f.fun, C.CUfunc_cache(config))
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (f *Function) SetAttribute(attr FunctionAttribute, value int) Result {
	stat := C.cuFuncSetAttribute(f.fun, C.CUfunction_attribute(attr), C.int(value))
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (f *Function) Launch(grid, block Dim3, args ...unsafe.Pointer) Result {
	return f.LaunchEx(grid, block, 0, nil, args...)
}

// TODO: add attributes
func (f *Function) LaunchEx(grid, block Dim3, sharedMem uint64, stream *Stream /*attributes?,*/, args ...unsafe.Pointer) Result {
	return internalLaunchEx(f.fun, grid, block, sharedMem, stream, args...)
}

func (f *Function) NativePointer() uintptr {
	return uintptr(unsafe.Pointer(f.fun))
}

func internalLaunchEx(func_ C.CUfunction, grid, block Dim3, sharedMem uint64, stream *Stream, args ...unsafe.Pointer) Result {
	offset := func(ptr unsafe.Pointer, off int, size uintptr) unsafe.Pointer {
		return unsafe.Pointer(uintptr(ptr) + size*uintptr(off))
	}
	config := &C.CUlaunchConfig{
		C.uint(grid.X),
		C.uint(grid.Y),
		C.uint(grid.Z),
		C.uint(block.X),
		C.uint(block.Y),
		C.uint(block.Z),
		C.uint(sharedMem), //sharedMemBytes
		nil,               //stream, 0 for default
		nil,               //CUlaunchAttribute*
		0,                 //numAttrs
		[4]byte{},         //___cgo alignment
	}

	if stream != nil {
		config.hStream = stream.stream
	}

	ptrSize := unsafe.Sizeof(uintptr(0))

	//TODO: imporve as you did others, into slices
	//copy of args to C
	argp := C.malloc(C.size_t(len(args)) * C.size_t(ptrSize))
	argv := C.malloc(C.size_t(len(args)) * C.size_t(ptrSize))
	defer C.free(argp)
	defer C.free(argv)

	for i := range args {
		*(*unsafe.Pointer)(offset(argp, i, ptrSize)) = offset(argv, i, ptrSize) // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i, ptrSize))) = *((*uint64)(args[i]))          // argv[i] = args[i]
	}

	stat := C.cuLaunchKernelEx(config, func_, (*unsafe.Pointer)(argp), nil)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

const (
	//Ignored entry, for convenient composition
	CU_LAUNCH_ATTRIBUTE_IGNORE LauncAttributeID = 0
	// Valid for streams, graph nodes, launches. See CUlaunchAttributeValue::accessPolicyWindow.
	CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW LauncAttributeID = 1
	// Valid for graph nodes, launches. See CUlaunchAttributeValue::cooperative.
	CU_LAUNCH_ATTRIBUTE_COOPERATIVE LauncAttributeID = 2
	// Valid for streams. See CUlaunchAttributeValue::syncPolicy.
	CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY LauncAttributeID = 3
	// Valid for graph nodes, launches. See CUlaunchAttributeValue::clusterDim.
	CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION LauncAttributeID = 4
	// Valid for graph nodes, launches. See CUlaunchAttributeValue::clusterSchedulingPolicyPreference.
	CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE LauncAttributeID = 5
	// Valid for launches. Setting CUlaunchAttributeValue::programmaticStreamSerializationAllowed to non-0 signals that the kernel will use programmatic means to resolve its stream dependency, so that the CUDA runtime should opportunistically allow the grid's execution to overlap with the previous kernel in the stream, if that kernel requests the overlap. The dependent launches can choose to wait on the dependency using the programmatic sync (cudaGridDependencySynchronize() or equivalent PTX instructions).
	CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION LauncAttributeID = 6
	// Valid for launches. Set CUlaunchAttributeValue::programmaticEvent to record the event. Event recorded through this launch attribute is guaranteed to only trigger after all block in the associated kernel trigger the event. A block can trigger the event through PTX launchdep.release or CUDA builtin function cudaTriggerProgrammaticLaunchCompletion(). A trigger can also be inserted at the beginning of each block's execution if triggerAtBlockStart is set to non-0. The dependent launches can choose to wait on the dependency using the programmatic sync (cudaGridDependencySynchronize() or equivalent PTX instructions). Note that dependents (including the CPU thread calling cuEventSynchronize()) are not guaranteed to observe the release precisely when it is released. For example, cuEventSynchronize() may only observe the event trigger long after the associated kernel has completed. This recording type is primarily meant for establishing programmatic dependency between device tasks. Note also this type of dependency allows, but does not guarantee, concurrent execution of tasks. The event supplied must not be an interprocess or interop event. The event must disable timing (i.e. must be created with the CU_EVENT_DISABLE_TIMING flag set).
	CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT LauncAttributeID = 7
	// Valid for streams, graph nodes, launches. See CUlaunchAttributeValue::priority.
	CU_LAUNCH_ATTRIBUTE_PRIORITY LauncAttributeID = 8
	// Valid for streams, graph nodes, launches. See CUlaunchAttributeValue::memSyncDomainMap.
	CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP LauncAttributeID = 9
	// Valid for streams, graph nodes, launches. See CUlaunchAttributeValue::memSyncDomain.
	CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN LauncAttributeID = 10
	// Valid for launches. Set CUlaunchAttributeValue::launchCompletionEvent to record the event. Nominally, the event is triggered once all blocks of the kernel have begun execution. Currently this is a best effort. If a kernel B has a launch completion dependency on a kernel A, B may wait until A is complete. Alternatively, blocks of B may begin before all blocks of A have begun, for example if B can claim execution resources unavailable to A (e.g. they run on different GPUs) or if B is a higher priority than A. Exercise caution if such an ordering inversion could lead to deadlock. A launch completion event is nominally similar to a programmatic event with triggerAtBlockStart set except that it is not visible to cudaGridDependencySynchronize() and can be used with compute capability less than 9.0. The event supplied must not be an interprocess or interop event. The event must disable timing (i.e. must be created with the CU_EVENT_DISABLE_TIMING flag set).
	CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT LauncAttributeID = 12
	// Valid for graph nodes, launches. This attribute is graphs-only, and passing it to a launch in a non-capturing stream will result in an error. CUlaunchAttributeValue::deviceUpdatableKernelNode::deviceUpdatable can only be set to 0 or 1. Setting the field to 1 indicates that the corresponding kernel node should be device-updatable. On success, a handle will be returned via CUlaunchAttributeValue::deviceUpdatableKernelNode::devNode which can be passed to the various device-side update functions to update the node's kernel parameters from within another kernel. For more information on the types of device updates that can be made, as well as the relevant limitations thereof, see cudaGraphKernelNodeUpdatesApply. Nodes which are device-updatable have additional restrictions compared to regular kernel nodes. Firstly, device-updatable nodes cannot be removed from their graph via cuGraphDestroyNode. Additionally, once opted-in to this functionality, a node cannot opt out, and any attempt to set the deviceUpdatable attribute to 0 will result in an error. Device-updatable kernel nodes also cannot have their attributes copied to/from another kernel node via cuGraphKernelNodeCopyAttributes. Graphs containing one or more device-updatable nodes also do not allow multiple instantiation, and neither the graph nor its instantiated version can be passed to cuGraphExecUpdate. If a graph contains device-updatable nodes and updates those nodes from the device from within the graph, the graph must be uploaded with cuGraphUpload before it is launched. For such a graph, if host-side executable graph updates are made to the device-updatable nodes, the graph must be uploaded before it is launched again.
	CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE LauncAttributeID = 13
	// Valid for launches. On devices where the L1 cache and shared memory use the same hardware resources, setting CUlaunchAttributeValue::sharedMemCarveout to a percentage between 0-100 signals the CUDA driver to set the shared memory carveout preference, in percent of the total shared memory for that kernel launch. This attribute takes precedence over CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT. This is only a hint, and the CUDA driver can choose a different configuration if required for the launch.
	CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT LauncAttributeID = 14
)
