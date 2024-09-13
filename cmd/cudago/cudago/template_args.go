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

type Arg struct {
	Name string
	Type string
}

type CuFileFunc struct {
	Name     string // Exported name
	RawName  string // Original name
	GoArgs   []Arg  // Go name -> Go type; arrays preserve ordering
	CArgs    []Arg  // C name -> C type; arrays preseve ordering, discards const, unsigned, and pointers
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
	k.FileName = strings.Trim(strings.Map(validMap, name), "_")
}

func (k *TemplateArgs) SetPackage(name string) {
	k.Package = strings.Trim(strings.Map(validMap, name), "_")
}

// SetArgs sets the C and Go names and types of the arguments into their respective maps
func (k *CuFileFunc) SetArgs(args []string) {
	k.CArgs = make([]Arg, len(args))
	k.GoArgs = make([]Arg, len(args))
	for i, arg := range args {
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
		//TODO: CArgs.Type should be the original type witha all modifiers (pointer, const, unsigned)
		k.GoArgs[i] = Arg{cArgName, targ}
		k.CArgs[i] = Arg{cArgName, cArgType}
	}
}

func validMap(r rune) rune {
	if (!unicode.IsLetter(r) && !unicode.IsDigit(r)) || unicode.IsSpace(r) {
		return '_'
	}
	return r
}

const unknownGoType = "unsafe.Pointer"
const goPointerType = "uintptr" // reserve: unsafe.Pointer

// key is the C type, value is the Go type
var typeMap = map[string]string{
	"int":    "int",
	"char":   "byte",
	"short":  "int16",
	"long":   "int64",
	"float":  "float32",
	"double": "float64",
}
