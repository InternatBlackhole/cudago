# Example: Add to all (previously named Increase by One)
This example demonstrates how to use the CUDA Go wrapper library and the CudaGo tool to generate wrappers of `.cu` files.

## How to run

**Warning:** Before you run the Go version, please make sure that you have followed all previous instructions to set up your environment.

To run do the following:
1. `cd` into this directory.
2. `CudaGo -precompile -package cu -- ints.cu`
3. `go mod tidy`.
4. `go run .`