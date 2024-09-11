package nvrtc

//#include "nvrtc.h"
import "C"
import "unsafe"

type Program struct {
	prog C.nvrtcProgram
}

type Header struct {
	SrcCode     string // The source code of the header. This is the code that will be included in the program.
	IncludeName string // The name of the header. This is the name that will be used to include the header in the program.
}

/*
CreateProgram creates a program object from the source code string.
The name parameter is used to specify the name of the program.

The headers parameter is used to specify the headers that will be included in the program.
They have to contain the source code of the header and the name of the header.
This can be omiited if the program does not have any headers or if the headers are in the filesystem and will be included wit -I flags when compiling
*/
func CreateProgram(srcCode string, name string, headers []Header) (*Program, error) {
	var prog C.nvrtcProgram
	cSrc := C.CString(srcCode)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cSrc))
	defer C.free(unsafe.Pointer(cName))
	cHeaders := make([]*C.char, len(headers))
	cIncludeNames := make([]*C.char, len(headers))
	for i, header := range headers {
		cHeaders[i] = C.CString(header.SrcCode)
		cIncludeNames[i] = C.CString(header.IncludeName)
		defer C.free(unsafe.Pointer(cHeaders[i]))
		defer C.free(unsafe.Pointer(cIncludeNames[i]))
	}
	err := C.nvrtcCreateProgram(&prog, cSrc, cName, C.int(len(headers)), &cHeaders[0], &cIncludeNames[0])
	if err != C.NVRTC_SUCCESS {
		return nil, NewNvRtcError(uint32(err))
	}
	return &Program{prog}, nil
}

func (p *Program) Destroy() error {
	err := C.nvrtcDestroyProgram(&p.prog)
	if err != C.NVRTC_SUCCESS {
		return NewNvRtcError(uint32(err))
	}
	return nil
}

/*
Compiles the program with the specified options.
Options are the same as the options that can be passed to nvcc (and gcc).
*/
func (p *Program) Compile(options []string) error {
	cOptions := make([]*C.char, len(options))
	for i, option := range options {
		cOptions[i] = C.CString(option)
		defer C.free(unsafe.Pointer(cOptions[i]))
	}
	err := C.nvrtcCompileProgram(p.prog, C.int(len(options)), &cOptions[0])
	if err != C.NVRTC_SUCCESS {
		return NewNvRtcError(uint32(err))
	}
	return nil
}

/*
Return the log from previous compilation
*/
func (p *Program) GetLog() (string, error) {
	var size C.size_t
	err := C.nvrtcGetProgramLogSize(p.prog, &size)
	if err != C.NVRTC_SUCCESS {
		return "", NewNvRtcError(uint32(err))
	}

	log := (*C.char)(C.malloc(size * C.sizeof_char))
	defer C.free(unsafe.Pointer(log))

	err = C.nvrtcGetProgramLog(p.prog, log)
	if err != C.NVRTC_SUCCESS {
		return "", NewNvRtcError(uint32(err))
	}

	return C.GoString(log), nil
}

func (p *Program) GetPTX() ([]byte, error) {
	var size C.size_t
	err := C.nvrtcGetPTXSize(p.prog, &size)
	if err != C.NVRTC_SUCCESS {
		return nil, NewNvRtcError(uint32(err))
	}

	data := C.malloc(size * C.sizeof_char)
	defer C.free(unsafe.Pointer(data))

	err = C.nvrtcGetPTX(p.prog, (*C.char)(data))
	if err != C.NVRTC_SUCCESS {
		return nil, NewNvRtcError(uint32(err))
	}

	return C.GoBytes(data, C.int(size)), nil
}
