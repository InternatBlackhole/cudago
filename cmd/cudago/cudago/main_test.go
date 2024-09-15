package main

import (
	"os"
	"strings"
	"testing"
)

func TestFileTemplateCreation(t *testing.T) {
	// Test the file template creation

	args := NewTemplateArgs()
	args.Package = "main"
	args.SetPTXCode("some ptx code")
	args.SetFileName("edges")
	args.Constants = map[string]string{"someConst": "int", "someOtherConst": "float"}
	args.Variables = map[string]string{"someVar": "float", "someOtherVar": "char"}

	fun := args.NewFunc()
	fun.SetName("borders")
	cArgs := strings.Split("unsigned char *origImage, int width", ",")

	for i, arg := range cArgs {
		cArgs[i] = strings.TrimSpace(arg)
	}

	fun.SetArgs(cArgs)
	fun.IsKernel = true

	args.AddFunc(fun)
	file := os.Stdout
	err := createProdFile(args, file)
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
