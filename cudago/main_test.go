package main

import (
	"testing"
)

const (
	testKernel     = `__global__ void borders(unsigned char *origImage, int width, int height, unsigned char *gradient, int imgSize)`
	testKernelName = "borders"
	testKernelArgs = "unsigned char *origImage, int width, int height, unsigned char *gradient, int imgSize"
)

var testKernelArgsArr = [...]string{"unsigned char *origImage", "int width", "int height", "unsigned char *gradient", "int imgSize"}

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

/*func TestFileTemplateCreation(t *testing.T) {
	// Test the file template creation
	kernel := NewKernel(testKernelName, testKernelArgsArr[:])
	file, err := createFileTemplate(kernel)
	if err != nil {
		t.Fatal(err)
	} else {
		t.Log("File template created successfully")
	}

	t.Log(file)
}*/
