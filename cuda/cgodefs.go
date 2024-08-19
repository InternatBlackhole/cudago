package cuda

//#cgo LDFLAGS:-lcuda
//
////default location:
//#cgo LDFLAGS:-L/usr/local/cuda/lib64/stubs/
//#cgo CFLAGS: -I/usr/local/cuda/include/
//
////WINDOWS:
//#cgo windows LDFLAGS:-LC:/cuda/lib/x64
//#cgo windows CFLAGS: -IC:/cuda/include
import "C"
