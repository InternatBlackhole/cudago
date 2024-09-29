package cudaruntime

/*
	This file is used to define the cgo flags for the cudaruntime package.

	If your path to cuda differs from the ubuntu default, please provide the correct CFLAGS and LDFLAGS
	through the CGO_CFLAGS and CGO_LDFLAGS environment varibles.
*/

////LINUX: default paths
//#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -lcudart
//#cgo linux CFLAGS: -I/usr/local/cuda/include
//
////WINDOWS:
//#cgo windows LDFLAGS:-LC:/cuda/lib/x64
//#cgo windows CFLAGS: -IC:/cuda/include
import "C"
