package main

// #cgo CFLAGS: -I/usr/local/cuda/include
// #include <stdio.h>
// #include <cuda.h>
import "C"
import (
	"flag"
	"os/exec"
	"regexp"
	"strings"
)

const (
	// kernelRegex is the regex to match a CUDA kernel function
	// format: __global__ void kernelName(type1 arg1, type2 arg2, ...)
	kernelRegex = `__global__\s+void\s+(\w+)\s*\(([^)]*)\)`
)

func main() {
	println(devTemplate)
	//get flags from the command line; this programs flags are until I encounter '--', then nvcc flags and params are after that
	flag.BoolVar(&isProd, "prod", false, "Set to true if you want to compile with production flags")

	flag.Parse()

	nvccFlags := flag.Args() //last ones are the file to search for kernels in
	//add my flags
	nvccFlags = append(nvccFlags, "--device-c", "-ptx")

	//first let's check if nvcc even wants to compile

	cmd := exec.Command("nvcc", nvccFlags...)
	err := cmd.Run()

	if err != nil {
		panic(err)
	}
}

func prodVersion() {
	//do something
}

func devVersion() {
	//do something
}

func getKernelNameAndArgs(kernel string) (string, []string, error) {
	kernelRegex, err := regexp.Compile(kernelRegex)

	if err != nil {
		return "", nil, err
	}

	kernelName := kernelRegex.FindStringSubmatch(kernel)

	splitArgs := strings.Split(kernelName[2], ",")
	for i, arg := range splitArgs {
		splitArgs[i] = strings.TrimSpace(arg)
	}

	return kernelName[1], splitArgs, nil
}

func usage() {
	//do something
}

var (
	isProd      = false
	devTemplate = `{{templates/dev.template}}` //FILL WITH FILE IN BRACKETS
)
