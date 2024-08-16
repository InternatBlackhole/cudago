package main

type Kernel struct {
	Name       string
	Args       []string
	CGoOptions []string
	Includes   []string
}

func NewKernel(name string, args []string) *Kernel {
	return &Kernel{
		Name:       name,
		Args:       args,
		CGoOptions: nil,
		Includes:   nil,
	}
}
