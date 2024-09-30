package cuda

//#include <cuda.h>
import "C"
import (
	"errors"
	"math"
)

var (
	ErrModuleNotFound = newStaticError(errors.New("module not found"))
	ErrDataIsEmtpy    = newStaticError(errors.New("data is empty"))
	ErrDeprecated     = newStaticError(errors.New("deprecated"))
	ErrUnsupported    = newStaticError(errors.ErrUnsupported)
	//ErrInvalidParam   = newStaticError(errors.New("invalid parameter"))
)

type Error struct {
	res C.CUresult
}

type internalError struct {
	error
}

// is also an error, Stringer
type Result interface {
	error
	Code() uint32
	ErrorString() string
	ErrorName() string
	String() string
}

func NewCudaError(err uint32) Result {
	return Error{C.CUresult(err)}
}

func newStaticError(err error) Result {
	return internalError{err}
}

func newInternalError(msg string) Result {
	return internalError{errors.New(msg)}
}

func (r Error) String() string {
	return r.Error()
}

func (r Error) Error() string {
	return r.ErrorName() + ": " + r.ErrorString()
}

func (r Error) Code() uint32 {
	return uint32(r.res)
}

func (r Error) ErrorString() string {
	var str *C.char
	err := C.cuGetErrorString(r.res, &str)
	if err != C.CUDA_SUCCESS {
		return "CUDA_ERROR_INVALID_VALUE"
	}
	return C.GoString(str)
}

func (r Error) ErrorName() string {
	var str *C.char
	err := C.cuGetErrorName(r.res, &str)
	if err != C.CUDA_SUCCESS {
		return "CUDA_ERROR_INVALID_VALUE"
	}
	return C.GoString(str)
}

func (r internalError) String() string {
	return r.ErrorString()
}

func (r internalError) Error() string {
	return r.ErrorName() + ": " + r.ErrorString()
}

func (r internalError) Code() uint32 {
	return math.MaxUint32
}

func (r internalError) ErrorString() string {
	return r.Error()
}

func (r internalError) ErrorName() string {
	return "InternalError"
}
