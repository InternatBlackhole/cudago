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
	var heads, includes **C.char = nil, nil
	if len(headers) > 0 {
		heads = &cHeaders[0]
		includes = &cIncludeNames[0]
	}
	err := C.nvrtcCreateProgram(&prog, cSrc, cName, C.int(len(headers)), heads, includes)
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
	if options == nil {
		options = make([]string, 0)
	}
	cOptions := make([]*C.char, len(options))
	for i, option := range options {
		cOptions[i] = C.CString(option)
		defer C.free(unsafe.Pointer(cOptions[i]))
	}
	var opts **C.char = nil
	if len(options) > 0 {
		opts = &cOptions[0]
	}
	err := C.nvrtcCompileProgram(p.prog, C.int(len(options)), opts)
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

	return C.GoBytes(data, C.int(size-1)), nil // -1 to remove the null terminator
}

func (p *Program) GetCubin() ([]byte, error) {
	var size C.size_t
	err := C.nvrtcGetCUBINSize(p.prog, &size)
	if err != C.NVRTC_SUCCESS {
		return nil, NewNvRtcError(uint32(err))
	}

	data := C.malloc(size * C.sizeof_char)
	defer C.free(unsafe.Pointer(data))

	err = C.nvrtcGetCUBIN(p.prog, (*C.char)(data))
	if err != C.NVRTC_SUCCESS {
		return nil, NewNvRtcError(uint32(err))
	}

	return C.GoBytes(data, C.int(size)), nil
}

func (p *Program) GetLTOIR() ([]byte, error) {
	var size C.size_t
	err := C.nvrtcGetLTOIRSize(p.prog, &size)
	if err != C.NVRTC_SUCCESS {
		return nil, NewNvRtcError(uint32(err))
	}

	data := C.malloc(size * C.sizeof_char)
	defer C.free(unsafe.Pointer(data))

	err = C.nvrtcGetLTOIR(p.prog, (*C.char)(data))
	if err != C.NVRTC_SUCCESS {
		return nil, NewNvRtcError(uint32(err))
	}

	return C.GoBytes(data, C.int(size)), nil
}

func (p *Program) GetOptiXIR() ([]byte, error) {
	var size C.size_t
	err := C.nvrtcGetOptiXIRSize(p.prog, &size)
	if err != C.NVRTC_SUCCESS {
		return nil, NewNvRtcError(uint32(err))
	}

	data := C.malloc(size * C.sizeof_char)
	defer C.free(unsafe.Pointer(data))

	err = C.nvrtcGetOptiXIR(p.prog, (*C.char)(data))
	if err != C.NVRTC_SUCCESS {
		return nil, NewNvRtcError(uint32(err))
	}

	return C.GoBytes(data, C.int(size)), nil
}

func (p *Program) AddNameExpression(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	err := C.nvrtcAddNameExpression(p.prog, cName)
	if err != C.NVRTC_SUCCESS {
		return NewNvRtcError(uint32(err))
	}
	return nil
}

func (p *Program) GetLoweredName(name string) (string, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var loweredName *C.char
	err := C.nvrtcGetLoweredName(p.prog, cName, &loweredName)
	if err != C.NVRTC_SUCCESS {
		return "", NewNvRtcError(uint32(err))
	}

	return C.GoString(loweredName), nil
}

//TODO: look into nvrtcGetTypeName
