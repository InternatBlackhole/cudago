package cuda

//#include <cuda.h>
import "C"
import "unsafe"

// Wrapper for CUDA Context
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX
type Context struct {
	ctx C.CUcontext
	//device *Device
}

// Wrapper for CUDA Primary Context

// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX
type PrimaryCtx struct {
	Context
	device *Device
}

// Flags for CUDA Contexts.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX
type ContextFlags uint32

// Limit number for CUDA Contexts.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8
type Limit int32

// Cache configuration for CUDA Contexts.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3
type CacheConfig uint32

// Create a new CUDA context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf
func NewContext(flags ContextFlags, device *Device) (*Context, Result) {
	var ctx C.CUcontext
	stat := C.cuCtxCreate(&ctx, C.uint(flags), device.dev)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Context{ctx}, nil
}

// Utility type for creating a new CUDA context with affinity parameters.
// Used in NewContext_v3.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2a5b565b1fb067f319c98787ddfa4016
type AffinityParam struct {
	Param AffinityType
	Value int
}

// Create a new CUDA context with execution affinity.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2a5b565b1fb067f319c98787ddfa4016
func NewContext_v3(flags ContextFlags, device *Device, params ...AffinityParam) (*Context, Result) {
	var ctx C.CUcontext
	var cparams []C.CUexecAffinityParam
	for _, p := range params {
		cparams = append(cparams, C.CUexecAffinityParam{C.CUexecAffinityType(p.Param), [4]byte(unsafe.Slice((*byte)(unsafe.Pointer(&p.Value)), 4))})
	}
	stat := C.cuCtxCreate_v3(&ctx, &cparams[0], C.int(len(cparams)), C.uint(flags), device.dev)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Context{ctx}, nil
}

// THIS FUNCTION IS CURRENTLY NOT IMPLEMENTED.
// DO NOT USE.
//
// Create a new CUDA context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gd84cbb0ad9470d66dc55e0830d56ef4d
func NewContext_v4() {
	// TODO: Implement cuCtxCreate_v4
}

// Gets the native handle of the CUDA context.
func (c *Context) NativePointer() uintptr {
	return uintptr(unsafe.Pointer(c.ctx))
}

// Destroy the CUDA context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e
func (c *Context) Destroy() Result {
	stat := C.cuCtxDestroy(c.ctx)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Get's the context's API version.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb
func (c *Context) GetApiVersion() (version uint32, err Result) {
	var _version C.uint
	stat := C.cuCtxGetApiVersion(c.ctx, &_version)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return uint32(_version), nil
}

// Returns the preferred cache configuration for the current context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360
func (c *Context) GetCacheConfig() (CacheConfig, Result) {
	var config C.CUfunc_cache
	stat := C.cuCtxGetCacheConfig(&config)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return CacheConfig(config), nil
}

// Sets the preferred cache configuration for the current context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3
func (c *Context) SetCacheConfig(config CacheConfig) Result {
	stat := C.cuCtxSetCacheConfig(C.CUfunc_cache(config))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Records an event.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gf3ee63561a7a371fa9d4dc0e31f94afd
func (c *Context) RecordEvent(event *Event) Result {
	stat := C.cuCtxRecordEvent(c.ctx, event.event)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

// Make context wait on an event.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gcf64e420275a8141b1f12bfce3f478f9
func (c *Context) WaitEvent(event *Event) Result {
	stat := C.cuCtxWaitEvent(c.ctx, event.event)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

// Returns the unique Id associated with the context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g32f492cd6c3f90af0d6935b294392db5
func (c *Context) GetId() (id uint64, err Result) {
	var _id C.ulonglong
	stat := C.cuCtxGetId(c.ctx, &_id)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return uint64(_id), nil
}

// Returns the device ID for the current context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e
func GetCurrentContextDevice() (*Device, Result) {
	var dev C.CUdevice
	stat := C.cuCtxGetDevice(&dev)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Device{dev}, nil
}

// Returns the execution affinity setting for the current context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g83421924a20536a4df538111cf61b405
func GetCurrentContextExecAffinity(adType AffinityType) (value int, err Result) {
	_value := C.CUexecAffinityParam{}
	stat := C.cuCtxGetExecAffinity(&_value, C.CUexecAffinityType(adType))

	if stat != C.CUDA_SUCCESS {
		return -1, NewCudaError(uint32(stat))
	}

	return *(*int)(unsafe.Pointer(&_value.param[0])), nil
}

// Returns the flags for the current context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d
func GetCurrentContextFlags() (ContextFlags, Result) {
	var flags C.uint
	stat := C.cuCtxGetFlags(&flags)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return ContextFlags(flags), nil
}

// Sets the flags for the current context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g66655c37602c8628eae3e40c82619f1e
func SetCurrentContextFlags(flags ContextFlags) Result {
	stat := C.cuCtxSetFlags(C.uint(flags))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Returns resource limits.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8
func GetCurrentContextLimit(limit Limit) (value uint64, err Result) {
	var _value C.size_t
	stat := C.cuCtxGetLimit(&_value, C.CUlimit(limit))

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return uint64(_value), nil
}

// Set resource limits.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a
func SetCurrentContextLimit(limit Limit, value uint64) Result {
	stat := C.cuCtxSetLimit(C.CUlimit(limit), C.size_t(value))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Returns numerical values that correspond to the least and greatest stream priorities.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091
func GetCurrentContextStreamPriorityRange() (low int, high int, err Result) {
	var _low, _high C.int
	stat := C.cuCtxGetStreamPriorityRange(&_low, &_high)

	if stat != C.CUDA_SUCCESS {
		return 0, 0, NewCudaError(uint32(stat))
	}

	return int(_low), int(_high), nil
}

// Returns the CUDA context bound to the calling CPU thread.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0
func GetCurrentContext() (*Context, Result) {
	var ctx C.CUcontext
	stat := C.cuCtxGetCurrent(&ctx)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Context{ctx}, nil
}

// Binds the specified CUDA context to the calling CPU thread.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7
func SetCurrentContext(ctx *Context) Result {
	stat := C.cuCtxSetCurrent(ctx.ctx)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Resets all persisting lines in cache to normal status.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gb529532b5b1aef808295a6d1d18a0823
func ResetCurrentContextPersistingL2Cache() Result {
	stat := C.cuCtxResetPersistingL2Cache()

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Block for the current context's tasks to complete.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616
func CurrentContextSynchronize() Result {
	stat := C.cuCtxSynchronize()

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Pushes a context on the current CPU thread.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba
func PushCurrentContext(ctx *Context) Result {
	stat := C.cuCtxPushCurrent(ctx.ctx)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Pops the current CUDA context from the current CPU thread.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902
func PopCurrentContext() (*Context, Result) {
	var ctx C.CUcontext
	stat := C.cuCtxPopCurrent(&ctx)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Context{ctx}, nil
}

// Retain the primary context on the GPU.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300
func DevicePrimaryCtxRetain(device *Device) (*PrimaryCtx, Result) {
	var ctx C.CUcontext
	stat := C.cuDevicePrimaryCtxRetain(&ctx, C.int(device.dev))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &PrimaryCtx{Context{ctx}, device}, nil
}

// Release the primary context on the GPU.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad
func (c *PrimaryCtx) Release() Result {
	stat := C.cuDevicePrimaryCtxRelease(C.int(c.device.dev))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Get the state of the primary context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g65f3e018721b6d90aa05cfb56250f469
func (c *PrimaryCtx) GetState() (ContextFlags, bool, Result) {
	var flags C.uint
	var active C.int
	stat := C.cuDevicePrimaryCtxGetState(C.int(c.device.dev), &flags, &active)

	if stat != C.CUDA_SUCCESS {
		return 0, false, NewCudaError(uint32(stat))
	}

	return ContextFlags(flags), active == 1, nil
}

// Set flags for the primary context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf
func (c *PrimaryCtx) SetFlags(flags ContextFlags) Result {
	stat := C.cuDevicePrimaryCtxSetFlags(c.device.dev, C.uint(flags))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Destroy all allocations and reset all state on the primary context.
//
// See: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g5d38802e8600340283958a117466ce12
func (c *PrimaryCtx) Reset() Result {
	stat := C.cuDevicePrimaryCtxReset(C.int(c.device.dev))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// Return the native handle of the primary context.
func (c *PrimaryCtx) NativeHandle() uintptr {
	return uintptr(unsafe.Pointer(c.ctx))
}

const (
	CU_LIMIT_STACK_SIZE                       Limit = 0x00 // stack size in bytes of each GPU thread.
	CU_LIMIT_PRINTF_FIFO_SIZE                 Limit = 0x01 // size in bytes of the FIFO used by the printf() device system call.
	CU_LIMIT_MALLOC_HEAP_SIZE                 Limit = 0x02 // size in bytes of the heap used by the malloc() and free() device system calls.
	CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH           Limit = 0x03 // maximum grid depth at which a thread can issue the device runtime call cudaDeviceSynchronize() to wait on child grid launches to complete.
	CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT Limit = 0x04 // maximum number of outstanding device runtime launches that can be made from this context.
	CU_LIMIT_MAX_L2_FETCH_GRANULARITY         Limit = 0x05 // L2 cache fetch granularity.
	CU_LIMIT_PERSISTING_L2_CACHE_SIZE         Limit = 0x06 // Persisting L2 cache size in bytes
	CU_LIMIT_SHMEM_SIZE                       Limit = 0x07 // A maximum size in bytes of shared memory available to CUDA kernels on a CIG context. Can only be queried, cannot be set
	CU_LIMIT_CIG_ENABLED                      Limit = 0x08 // A non-zero value indicates this CUDA context is a CIG-enabled context. Can only be queried, cannot be set
	CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED       Limit = 0x09 // When set to a non-zero value, CUDA will fail to launch a kernel on a CIG context, instead of using the fallback path, if the kernel uses more shared memory than available
	CU_LIMIT_MAX                              Limit = 0x0A
)

const (
	CU_FUNC_CACHE_PREFER_NONE   CacheConfig = 0x00 // no preference for shared memory or L1 (default)
	CU_FUNC_CACHE_PREFER_SHARED CacheConfig = 0x01 // prefer larger shared memory and smaller L1 cache
	CU_FUNC_CACHE_PREFER_L1     CacheConfig = 0x02 // prefer larger L1 cache and smaller shared memory
	CU_FUNC_CACHE_PREFER_EQUAL  CacheConfig = 0x03 // prefer equal sized L1 cache and shared memory
	CU_FUNC_CACHE_MAX           CacheConfig = 0x04
)

const (
	CU_CTX_SCHED_AUTO           ContextFlags = 0x00 // Automatic scheduling
	CU_CTX_SCHED_SPIN           ContextFlags = 0x01 // Set spin as default scheduling
	CU_CTX_SCHED_YIELD          ContextFlags = 0x02 // Set yield as default scheduling
	CU_CTX_SCHED_BLOCKING_SYNC  ContextFlags = 0x04 // Set blocking synchronization as default scheduling
	CU_CTX_BLOCKING_SYNC        ContextFlags = 0x04 // Deprecated. This flag was deprecated as of CUDA 4.0 and was replaced with CU_CTX_SCHED_BLOCKING_SYNC. Set blocking synchronization as default scheduling
	CU_CTX_SCHED_MASK           ContextFlags = 0x07
	CU_CTX_MAP_HOST             ContextFlags = 0x08 // Deprecated. This flag was deprecated as of CUDA 11.0 and it no longer has any effect. All contexts as of CUDA 3.2 behave as though the flag is enabled.
	CU_CTX_LMEM_RESIZE_TO_MAX   ContextFlags = 0x10 // Keep local memory allocation after launch
	CU_CTX_COREDUMP_ENABLE      ContextFlags = 0x20 // Trigger coredumps from exceptions in this context
	CU_CTX_USER_COREDUMP_ENABLE ContextFlags = 0x40 // Enable user pipe to trigger coredumps in this context
	CU_CTX_SYNC_MEMOPS          ContextFlags = 0x80 // Ensure synchronous memory operations on this context will synchronize
	CU_CTX_FLAGS_MASK           ContextFlags = 0xFF
)
