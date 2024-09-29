package cuda

/*
	This file is used to define the cgo flags for the cuda package.
	Since your system may have a different path to the cuda libraries, you may need to change the paths in the cgo flags.
	If your pkg-config is not set up correctly, you may need to change the flags to the correct paths.
	For example, you may need to add the version of the cuda library to the pkg-config flag.
	To solve this, you can make a cuda.pc file in /usr/lib/pkgconfig/ (or in PKG_CONFIG_PATH) (most simple way is to just copy the file and rename it).
	Or you can change the cgo flags to the correct paths. LD_FLAGS, CFLAGS, and the -l flags are needed.

	If your path to cuda differs from the ubuntu default please provide the correct CFLAGS and LDFLAGS
	through the CGO_CFLAGS and CGO_LDFLAGS environment varibles.
*/

//TODO: elimate usage of pkg-config and try to replace with env vars

////LINUX: default paths
//#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -lcuda
//#cgo linux CFLAGS: -I/usr/local/cuda/include
//
////WINDOWS:
//#cgo windows LDFLAGS:-LC:/cuda/lib/x64
//#cgo windows CFLAGS: -IC:/cuda/include
import "C"

type NativeHandle interface {
	NativePointer() uintptr
}
