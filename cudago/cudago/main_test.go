package main

import (
	"os"
	"strings"
	"testing"
)

const (
	testKernel     = `__global__ void borders(unsigned char *origImage, int width, int height, unsigned char *gradient, int imgSize)`
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

	kernel := NewKernel()
	kernel.SetName(testKernelName)
	kernel.Args = strings.Split(testKernelArgs, ",")

	for i, arg := range kernel.Args {
		kernel.Args[i] = strings.TrimSpace(arg)
	}

	kernel.SetPackage("main")
	file := os.Stdout
	err := createFileFromDevTemplate(kernel, file)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log("File template created successfully")
	}
}
