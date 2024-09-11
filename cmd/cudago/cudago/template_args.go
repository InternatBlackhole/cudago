package main

import (
	"strings"
	"unicode"
)

type TemplateArgs struct {
	Package string
	Funcs   []*CuFileFunc
	PTXCode string
}

type CuFileFunc struct {
	Name     string
	RawName  string
	GoArgs   map[string]string // arg name -> type
	CArgs    map[string]string // arg name -> type
	IsKernel bool
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
