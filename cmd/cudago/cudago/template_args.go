package main

import (
	"strings"
	"unicode"
)

type TemplateArgs struct {
	Package   string
	FileName  string
	Funcs     []*CuFileFunc
	Constants map[string]string // const name -> const type (is always cuda.MemAllocation)
	Variables map[string]string // var name -> var type (is always cuda.MemAllocation)
	PTXCode   string
}

type CuFileFunc struct {
	Name     string            // Exported name
	RawName  string            // Original name
	GoArgs   map[string]string // arg name -> type
	CArgs    map[string]string // arg name -> type
	IsKernel bool
}

type CuVar struct {
	Name      string
	CType     string
	DevicePtr string // string representation of uintptr
	Size      string // string representation of uint64
}

func NewTemplateArgs() *TemplateArgs {
	return &TemplateArgs{}
}

func NewTemplateFunc() *CuFileFunc {
	return &CuFileFunc{}
}

func (k *CuFileFunc) SetName(name string) {
	// Capitalize the first letter of the name
	k.RawName = name
	k.Name = string(unicode.ToUpper(rune(name[0]))) + name[1:]
}

func (k *TemplateArgs) SetPTXCode(code string) {
	k.PTXCode = code
}

func (k *TemplateArgs) AddFunc(f *CuFileFunc) {
	k.Funcs = append(k.Funcs, f)
}

func (k *TemplateArgs) SetFileName(name string) {
	valid := func(r rune) rune {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			return '_'
		}
		return r
	}
	k.FileName = strings.Map(valid, name)
}

// SetArgs sets the C and Go names and types of the arguments into their respective maps
func (k *CuFileFunc) SetArgs(args []string) {
	k.CArgs = make(map[string]string, len(args))
	k.GoArgs = make(map[string]string, len(args))
	for _, arg := range args {
		isUnsigned := false
		split := strings.Split(arg, " ")

		if split[0] == "const" {
			split = split[1:] // remove const
		}

		if split[0] == "unsigned" {
			isUnsigned = true
			split = split[1:] // remove unsigned
		}

		cArgType := strings.Join(split[:len(split)-1], " ")
		cArgName := strings.Trim(split[len(split)-1], "*")

		targ := typeMap[cArgType]
		if targ == "" {
			targ = unknownGoType
		}

		if strings.Contains(arg, "*") {
			targ = goPointerType
		} else if isUnsigned {
			targ = "u" + targ
		}

		k.GoArgs[cArgName] = targ
		k.CArgs[cArgName] = cArgType
	}
}

const unknownGoType = "unsafe.Pointer"
const goPointerType = "unsafe.Pointer"

// key is the C type, value is the Go type
var typeMap = map[string]string{
	"int":    "int",
	"char":   "int8",
	"short":  "int16",
	"long":   "int64",
	"float":  "float32",
	"double": "float64",
}
