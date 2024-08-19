package main

import (
	"unicode"
)

type Kernel struct {
	Name string
	Args []string
	//CGoOptions []string
	//Includes   []string
	Package string
	PTXCode []byte
}

func NewKernel() *Kernel {
	return &Kernel{}
}

/*func (k *Kernel) AddCGoOptions(options ...string) {
	k.CGoOptions = append(k.CGoOptions, options...)
}*/

/*func (k *Kernel) AddIncludes(includes ...string) {
	k.Includes = append(k.Includes, includes...)
}*/

func (k *Kernel) SetPackage(_package string) {
	k.Package = _package
}

func (k *Kernel) SetName(name string) {
	// Capitalize the first letter of the name
	k.Name = string(unicode.ToUpper(rune(name[0]))) + name[1:]
}

func (k *Kernel) SetPTXCode(code []byte) {
	k.PTXCode = code
}
