package nvrtc

/*
	This file is used to define the cgo flags for the nvrtc package.
	Since your system may have a different path to the nvrtc libraries, you may need to change the paths in the cgo flags.
	If your pkg-config is not set up correctly, you may need to change the flags to the correct paths.
	For example, you may need to add the version of the nvrtc library to the pkg-config flag.
	To solve this, you can make a nvrtc.pc file in /usr/lib/pkgconfig/ (or in PKG_CONFIG_PATH env var) (most simple way is to just copy the file and rename it).
	Or you can change the cgo flags to the correct paths. LD_FLAGS, CFLAGS, and the -l flags are needed.
*/

//#cgo pkg-config: nvrtc
//#cgo LDFLAGS: -lnvrtc
//#include "nvrtc.h"
//#include <stdlib.h>
import "C"
