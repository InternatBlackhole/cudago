# Example: GPU Capabilites
This example demonstrates how to use the CUDA Go wrapper library and the native library to print some GPU capabilities.

## How to run

**Warning:** Before you run the Go version, please make sure that you have followed all previous instructions to set up your environment.

To run the CUDA Go wrapper version, do the following:
1. `cd` into this directory.
2. `go mod tidy`.
3. `go run .`

To run the native version:
1. Make sure you have CUDA Toolkit installed on your system.
2. `nvcc -o cudainfo cuda_info.cu`
3. `./cudainfo`