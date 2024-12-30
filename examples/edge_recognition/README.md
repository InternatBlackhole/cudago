# Example: Edge recognition
This example demonstrates how to use the CUDA Go wrapper library and the CudaGo tool to generate wrappes for `.cu` files.

This example takes a `test.jpg` image in the same directory, and outputs an `output.jpg` image in black and white, where white represents the edges of input image. (Details about which algorithm is used are purposfully omitted).

## How to run

**Warning:** Before you run the Go version, please make sure that you have followed all previous instructions to set up your environment.

To run do the following:
1. `cd` into this directory.
1. `CudaGo -precompile -package cuda_stuff -- edges.cu`
1. Put some jpg image named `test.jpg` into this directory.
1. `go mod tidy`.
1. `go run .`
1. Look at `image.jpg`