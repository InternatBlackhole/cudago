package cuda

import "C"

/*type LinkState struct {
	state C.CUlinkState
}*/

//TODO: Implement if needed
/*
func NewCudaLinkState(linkOptions []JitOption) *CudaLinkState {
	var state C.CUlinkState
	opts := make([]C.CUjit_option, len(linkOptions))
	vals := make([]unsafe.Pointer, len(linkOptions))

	for i, opt := range linkOptions {
		opts[i] = C.CUjit_option(opt.Name)
		vals[i] = opt.Value
	}
}*/
