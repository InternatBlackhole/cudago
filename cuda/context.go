package cuda

//#include <cuda.h>
import "C"
import "unsafe"

type Context struct {
	ctx C.CUcontext
	//device *Device
}

type PrimaryCtx struct {
	Context
	device *Device
}

type ContextFlags uint32
type Limit int32
type CacheConfig uint32

func NewContext(flags ContextFlags, device *Device) (*Context, Result) {
	var ctx C.CUcontext
	stat := C.cuCtxCreate(&ctx, C.uint(flags), device.dev)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Context{ctx}, nil
}

type AffinityParam struct {
	Param AffinityType
	Value int
}

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

func NewContext_v4() {
	// TODO: Implement cuCtxCreate_v4
}

func (c *Context) NativePointer() uintptr {
	return uintptr(unsafe.Pointer(c.ctx))
}

func (c *Context) Destroy() Result {
	stat := C.cuCtxDestroy(c.ctx)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (c *Context) GetApiVersion() (version uint32, err Result) {
	var _version C.uint
	stat := C.cuCtxGetApiVersion(c.ctx, &_version)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return uint32(_version), nil
}

func (c *Context) GetCacheConfig() (CacheConfig, Result) {
	var config C.CUfunc_cache
	stat := C.cuCtxGetCacheConfig(&config)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return CacheConfig(config), nil
}

func (c *Context) SetCacheConfig(config CacheConfig) Result {
	stat := C.cuCtxSetCacheConfig(C.CUfunc_cache(config))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (c *Context) RecordEvent(event *Event) Result {
	stat := C.cuCtxRecordEvent(c.ctx, event.event)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (c *Context) WaitEvent(event *Event) Result {
	stat := C.cuCtxWaitEvent(c.ctx, event.event)
	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}

func (c *Context) GetId() (id uint64, err Result) {
	var _id C.ulonglong
	stat := C.cuCtxGetId(c.ctx, &_id)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return uint64(_id), nil
}

func GetCurrentContextDevice() (*Device, Result) {
	var dev C.CUdevice
	stat := C.cuCtxGetDevice(&dev)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Device{dev}, nil
}

func GetCurrentContextExecAffinity(adType AffinityType) (value int, err Result) {
	_value := C.CUexecAffinityParam{}
	stat := C.cuCtxGetExecAffinity(&_value, C.CUexecAffinityType(adType))

	if stat != C.CUDA_SUCCESS {
		return -1, NewCudaError(uint32(stat))
	}

	return *(*int)(unsafe.Pointer(&_value.param[0])), nil
}

func GetCurrentContextFlags() (ContextFlags, Result) {
	var flags C.uint
	stat := C.cuCtxGetFlags(&flags)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return ContextFlags(flags), nil
}

func SetCurrentContextFlags(flags ContextFlags) Result {
	stat := C.cuCtxSetFlags(C.uint(flags))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func GetCurrentContextLimit(limit Limit) (value uint64, err Result) {
	var _value C.size_t
	stat := C.cuCtxGetLimit(&_value, C.CUlimit(limit))

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return uint64(_value), nil
}

func SetCurrentContextLimit(limit Limit, value uint64) Result {
	stat := C.cuCtxSetLimit(C.CUlimit(limit), C.size_t(value))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func GetCurrentContextStreamPriorityRange() (low int, high int, err Result) {
	var _low, _high C.int
	stat := C.cuCtxGetStreamPriorityRange(&_low, &_high)

	if stat != C.CUDA_SUCCESS {
		return 0, 0, NewCudaError(uint32(stat))
	}

	return int(_low), int(_high), nil
}

func GetCurrentContext() (*Context, Result) {
	var ctx C.CUcontext
	stat := C.cuCtxGetCurrent(&ctx)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Context{ctx}, nil
}

func SetCurrentContext(ctx *Context) Result {
	stat := C.cuCtxSetCurrent(ctx.ctx)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func ResetCurrentContextPersistingL2Cache() Result {
	stat := C.cuCtxResetPersistingL2Cache()

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func CurrentContextSynchronize() Result {
	stat := C.cuCtxSynchronize()

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func PushCurrentContext(ctx *Context) Result {
	stat := C.cuCtxPushCurrent(ctx.ctx)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func PopCurrentContext() (*Context, Result) {
	var ctx C.CUcontext
	stat := C.cuCtxPopCurrent(&ctx)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &Context{ctx}, nil
}

func DevicePrimaryCtxRetain(device *Device) (*PrimaryCtx, Result) {
	var ctx C.CUcontext
	stat := C.cuDevicePrimaryCtxRetain(&ctx, C.int(device.dev))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &PrimaryCtx{Context{ctx}, device}, nil
}

func (c *PrimaryCtx) Release() Result {
	stat := C.cuDevicePrimaryCtxRelease(C.int(c.device.dev))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (c *PrimaryCtx) GetState() (ContextFlags, bool, Result) {
	var flags C.uint
	var active C.int
	stat := C.cuDevicePrimaryCtxGetState(C.int(c.device.dev), &flags, &active)

	if stat != C.CUDA_SUCCESS {
		return 0, false, NewCudaError(uint32(stat))
	}

	return ContextFlags(flags), active == 1, nil
}

func (c *PrimaryCtx) SetFlags(flags ContextFlags) Result {
	stat := C.cuDevicePrimaryCtxSetFlags(c.device.dev, C.uint(flags))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (c *PrimaryCtx) Reset() Result {
	stat := C.cuDevicePrimaryCtxReset(C.int(c.device.dev))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

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
