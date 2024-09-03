package cuda

//#include <cuda.h>
import "C"

type CudaContext struct {
	ctx    C.CUcontext
	device *CudaDevice
}

type CudaPrimaryCtx struct {
	CudaContext
}

type CudaContextFlags uint32

func NewContext(flags CudaContextFlags, device *CudaDevice) (*CudaContext, error) {
	var ctx C.CUcontext
	stat := C.cuCtxCreate(&ctx, C.uint(flags), device.dev)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &CudaContext{ctx, device}, nil
}

func GetCurrentContext() (*CudaContext, error) {
	var ctx C.CUcontext
	stat := C.cuCtxGetCurrent(&ctx)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &CudaContext{ctx, nil}, nil
}

func (c *CudaContext) Destroy() error {
	stat := C.cuCtxDestroy(c.ctx)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (c *CudaContext) SetCurrent() error {
	stat := C.cuCtxSetCurrent(c.ctx)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func ContextSynchronize() error {
	stat := C.cuCtxSynchronize()

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func DevicePrimaryCtxRetain(device *CudaDevice) (*CudaPrimaryCtx, error) {
	var ctx C.CUcontext
	stat := C.cuDevicePrimaryCtxRetain(&ctx, C.int(device.dev))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &CudaPrimaryCtx{CudaContext{ctx, device}}, nil
}

func (c *CudaPrimaryCtx) Release() error {
	stat := C.cuDevicePrimaryCtxRelease(C.int(c.device.dev))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (c *CudaPrimaryCtx) GetState() (CudaContextFlags, bool, error) {
	var flags C.uint
	var active C.int
	stat := C.cuDevicePrimaryCtxGetState(C.int(c.device.dev), &flags, &active)

	if stat != C.CUDA_SUCCESS {
		return 0, false, NewCudaError(uint32(stat))
	}

	return CudaContextFlags(flags), active == 1, nil
}

func (c *CudaPrimaryCtx) SetFlags(flags CudaContextFlags) error {
	stat := C.cuDevicePrimaryCtxSetFlags(c.device.dev, C.uint(flags))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (c *CudaPrimaryCtx) Reset() error {
	stat := C.cuDevicePrimaryCtxReset(C.int(c.device.dev))

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

const (
	CU_CTX_SCHED_AUTO           CudaContextFlags = 0x00 // Automatic scheduling
	CU_CTX_SCHED_SPIN           CudaContextFlags = 0x01 // Set spin as default scheduling
	CU_CTX_SCHED_YIELD          CudaContextFlags = 0x02 // Set yield as default scheduling
	CU_CTX_SCHED_BLOCKING_SYNC  CudaContextFlags = 0x04 // Set blocking synchronization as default scheduling
	CU_CTX_BLOCKING_SYNC        CudaContextFlags = 0x04 // Deprecated. This flag was deprecated as of CUDA 4.0 and was replaced with CU_CTX_SCHED_BLOCKING_SYNC. Set blocking synchronization as default scheduling
	CU_CTX_SCHED_MASK           CudaContextFlags = 0x07
	CU_CTX_MAP_HOST             CudaContextFlags = 0x08 // Deprecated. This flag was deprecated as of CUDA 11.0 and it no longer has any effect. All contexts as of CUDA 3.2 behave as though the flag is enabled.
	CU_CTX_LMEM_RESIZE_TO_MAX   CudaContextFlags = 0x10 // Keep local memory allocation after launch
	CU_CTX_COREDUMP_ENABLE      CudaContextFlags = 0x20 // Trigger coredumps from exceptions in this context
	CU_CTX_USER_COREDUMP_ENABLE CudaContextFlags = 0x40 // Enable user pipe to trigger coredumps in this context
	CU_CTX_SYNC_MEMOPS          CudaContextFlags = 0x80 // Ensure synchronous memory operations on this context will synchronize
	CU_CTX_FLAGS_MASK           CudaContextFlags = 0xFF
)
