package main

import (
	"os"
	"strings"
	"testing"
)

const (
	testKernel     = `extern "C" __global__ void borders(unsigned char *origImage, int width, int height, unsigned char *gradient, int imgSize)`
	testKernelName = "borders"
	testKernelArgs = "unsigned char *origImage, int width, int height, unsigned char *gradient, int imgSize"

	kernelArrParamTest = "__global__ void params(float A[N][N], float B[N][N], float C[N][N], float alpha, float beta, float **params)"
)

var testKernelArgsArr = [...]string{"unsigned char *origImage", "int width", "int height", "unsigned char *gradient", "int imgSize"}
var kernelArrParamTestArgs = [...]string{"float A[N][N]", "float B[N][N]", "float C[N][N]", "float alpha", "float beta", "float **params"}

func TestKernelNameRegexExtraction(t *testing.T) {
	// Test the kernelRegex
	name, _, err := getKernelNameAndArgs(testKernel)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log("Kernel regex compiled successfully")
	}

	if name != testKernelName {
		t.Fatalf("Kernel regex failed to match kernel: %s, got: %s", testKernel, name)
	} else {
		t.Log("Kernel regex matched kernel successfully")
	}

}

func TestKernelArgsRegexExtraction(t *testing.T) {
	// Test the argsRegex
	_, args, err := getKernelNameAndArgs(testKernel)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log("Args regex compiled successfully")
	}

	for i, arg := range args {
		if arg != testKernelArgsArr[i] {
			t.Fatalf("Args regex failed to match arg: %s, got: %s", testKernelArgsArr[i], arg)
		}
	}

}

func TestKernelArgsRegexExtractionArr(t *testing.T) {
	// Test the argsRegex
	_, args, err := getKernelNameAndArgs(kernelArrParamTest)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log("Args regex compiled successfully")
	}

	for i, arg := range args {
		if arg != kernelArrParamTestArgs[i] {
			t.Fatalf("Args regex failed to match arg: %s, got: %s", kernelArrParamTestArgs[i], arg)
		}
	}

}

func TestFileTemplateCreation(t *testing.T) {
	// Test the file template creation

	args := NewTemplateArgs()
	args.Package = "main"
	args.SetPTXCode("some ptx code")

	fun := NewTemplateFunc()
	fun.SetName(testKernelName)
	cArgs := strings.Split(testKernelArgs, ",")

	for i, arg := range cArgs {
		cArgs[i] = strings.TrimSpace(arg)
	}

	fun.SetArgs(cArgs)
	fun.IsKernel = true

	args.AddFunc(fun)
	file := os.Stdout
	err := createFileFromDevTemplate(args, file)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log("File template created successfully")
	}
}

/*func TestMultiKernel(t *testing.T) {
	// Test the file template creation
	inFile, err := os.OpenFile("../../../tests/multi-kernel.cu", os.O_RDONLY, 0644)
	if err != nil {
		t.Fatal(err)
	}

	kernels, err := getKernels(inFile)

	args := NewTemplateArgs()
	args.Package = "main"

	fun := NewTemplateFunc()
	fun.SetName(testKernelName)
	cArgs := strings.Split(testKernelArgs, ",")

	for i, arg := range cArgs {
		cArgs[i] = strings.TrimSpace(arg)
	}

	fun.SetArgs(cArgs)
	fun.IsKernel = true
	fun.SetPTXCode("some ptx code")

	args.AddFunc(fun)

	fun2 := NewTemplateFunc()
	fun2.SetName("someOtherKernel")
	cArgs2 := strings.Split(testKernelArgs, ",")

	for i, arg := range cArgs2 {
		cArgs2[i] = strings.TrimSpace(arg)
	}

	fun2.SetArgs(cArgs2)
	fun2.IsKernel = true
	fun2.SetPTXCode("some ptx code")

	args.AddFunc(fun2)

	file := os.Stdout
	err := createFileFromDevTemplate(args, file)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log("File template created successfully")
	}
}*/
