package cuda

//#include <cuda.h>
import "C"
import "runtime"

var (
	isCudaInitialized bool = false
)

// Represents the initialization options performed by the Init function.
type InitToken struct {
	DeviceIndex int
	Device      *Device
	PrimaryCtx  *PrimaryCtx
}

// Initializes the CUDA driver API for the current process,
// locks the current goroutine to it's OS thread (using runtime.LockOSThread),
// and binds the devices primary context to the current thread.
// This is the recommended way to initialize the CUDA driver API.
func Init(device int) (*InitToken, Result) {
	runtime.LockOSThread()
	err := DriverInit()
	if err != nil {
		return nil, err
	}
	dev, err := DeviceGet(device)
	if err != nil {
		return nil, err
	}
	pctx, err := DevicePrimaryCtxRetain(dev)
	if err != nil {
		return nil, err
	}
	err = SetCurrentContext(&pctx.Context)
	if err != nil {
		return nil, err
	}
	return &InitToken{device, dev, pctx}, nil
}

// Releases the primary context of the device and unlocks the current goroutine from it's OS thread.
// Call when you don't need to use the cuda library anymore.
func (token *InitToken) Close() {
	token.PrimaryCtx.Release()
	token.Device = nil
	token.PrimaryCtx = nil
	runtime.UnlockOSThread()
}

// Initialize the CUDA driver API.
// Initializes the driver API and must be called before any other function from the driver API in the current process.
// If DriverInit() (or Init()) has not been called, any function from the driver API will return CUDA_ERROR_NOT_INITIALIZED.
// Only use this function if you know what you are doing.
func DriverInit() Result {
	if isCudaInitialized {
		return nil
	}
	err := C.cuInit(0)
	if err != C.CUDA_SUCCESS {
		return NewCudaError(uint32(err))
	}
	isCudaInitialized = true
	return nil
}

// Returns the latest CUDA version supported by driver.
func DriverVersion() (int32, Result) {
	var version int32
	err := C.cuDriverGetVersion((*C.int)(&version))
	if err != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(err))
	}
	return version, nil
}
