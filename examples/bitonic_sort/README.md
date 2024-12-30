# Example: Bitonic Sort (with shared memory)
This example demonstrates how to use the CUDA Go wrapper library and the CudaGo tool to generate wrappers of `.cu` files.

This example is an implematation of bitonic sort in CUDA, using shared memory. It shows how to use the advanced function wrapper that CudaGo also generates.

## How to run

**Warning:** Before you run the Go version, please make sure that you have followed all previous instructions to set up your environment.

To run do the following:
1. `cd` into this directory.
2. `CudaGo -precompile -package cu -- bitonic-shared-mem.cu`
3. `go mod tidy`.
4. `go run .`