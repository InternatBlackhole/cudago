# CUDA GO

CUDA GO is a set of wrappers for the CUDA library in GO that also can
generate wrappers for CUDA `.cu` files, such that their kernels can easily be executed in GO.
It wraps the most commonly used functions from the CUDA driver API, and (in the future) the runtime API,
well as nvrtc (NVIDIA Runtime Compiler) in full.

## CUDA GO usage

Anyone can freely use the CUDA wrappers in contained in packages `cuda`, `cuda-runtime`, and `nvrtc` in their own projects.
They're available at `github.com/InternatBlackhole/cudago/<package name>`.

To use the wrapper generator you need to download the program contained in [releases](https://github.com/InternatBlackhole/cudago/releases)
or you can compile it yourself. If you opt for the latter, please refer to the [Compilation](#compilation-of-cudago) section.

The program supports two types of wrappers. They are called [Dev mode](#dev-mode-vs-release-mode--performance-mode) and [Release mode](#dev-mode-vs-release-mode--performance-mode).

Base usage is as follows:

```bash
cudago [flags] -- <files to compile>
```

You can specify the following flags:

```bash
Usage of ./cudago:
  -nvcc string
        Flags to pass to nvcc/nvrtc
  -package string
        Package name for the generated code and output directory
  -prod
        Set to true if you want to compile with production flags
Specify files to compile after '--'
```

**Caution:**
When compiling all Go packages that use any of the wrappers generated by cudago or in this repository, you need to set the `CGO_CFLAGS` (contains the path to CUDA include headers) and `CGO_LDFLAGS` (contains the path to CUDA compiled libraries) environment variables to include the CUDA libraries and headers.
The easiest way to do this is to use `pkg-config`:

```bash
export CGO_CFLAGS=$(pkg-config --cflags cuda-12.6) # or any other version
export CGO_LDFLAGS=$(pkg-config --libs cuda-12.6) # or any other version
```

### Limitations for CUDA .cu files

For the wrapper to correctly work all kernels need to compiled as C linkage. Thus all kernels need to be wrapped in a `extern "C" {}` block
or have `extern "C"` in front of the kernel declaration.

All kernels also have to be top level standalone functions. Currently the wrapper does not support kernels that are part of a class or a struct
or are templates.

### Dev mode vs Release mode / Performance mode

The difference is that the debug mode does (onetime) runtime compilation of the CUDA code, while the release mode compiles the CUDA code at compile time (when running the tool).
By using the debug mode, you can easily change the CUDA code without recompiling the GO code, just keep the file at the same location and don't add any new kernels or modify the existing ones. This also impacts performance, since the first inovaction of any kernel will compile code.
If this is not desired, use the release mode.

### Prerequisites

- Installed CUDA 12.6 (or higher) on system you intend to use the library wrappers in
- Installed Go version 1.20 or higher
- pkg-config installed on your system (recommended)

### Compilation of cudago

***Note:*** Currently the only supported platform is Linux. Windows and MacOS compilation hasn't been tested. You are welcome to try it out and open a pull request to make it work.

To compile the cudago tool, you need to have Go installed on your system.
Since the tool uses a wrapper of NVRTC, CUDA is also required.
You need to export or specify the `CGO_CFLAGS` (contains the path to CUDA compiled libraries) and `CGO_LDFLAGS` (contains the path to CUDA compiled libraries) environment variables when compiling.
The easiest way to do this is to use `pkg-config`:

```bash
export CGO_CFLAGS=$(pkg-config --cflags cuda-12.6) # or any other version
export CGO_LDFLAGS=$(pkg-config --libs cuda-12.6) # or any other version
```

Then just run the `go build` command with all your desired flags.
