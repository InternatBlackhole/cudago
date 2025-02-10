package main

import (
	"fmt"
	"os"
	"time"
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
)

const (
	//form     = "%s;%d;%d\n"
	//formHead = "Func;Time;RunNum"
	form     = "%s;%f\n"
	formHead = "Operation;Time"
)

func report(operation string, time time.Duration) {
	fmt.Printf(form, operation, float64(time.Nanoseconds()/1000)) //microseconds
}

func main() {
	os.Exit(mainWithExit())
}

func mainWithExit() int {
	fmt.Println(formHead)

	var err error

	dev, err := cuda.Init(0)
	panicErr(err)
	defer dev.Close()
	log("Device init")

	state := generatorState{}
	tests := []generatorParam{
		{"CopyNoReg", 10, func() { testNoReg(100_000_000) }},   //800MB
		{"CopyYesReg", 10, func() { testYesReg(100_000_000) }}, //800MB
	}

	gen := generator(&state, tests...)
	log("Starting tests")
	for fun := gen(); fun != nil; fun = gen() {
		fun()
	}
	log("Tests done")
	return 0
}

func testNoReg(arrLen int) {
	start := time.Now()
	arr := make([]int, arrLen)
	took := time.Since(start)
	report("HostMallocNormal", took)

	for i := 0; i < arrLen; i++ {
		arr[i] = i
	}
	arrByteSize := uint64(uintptr(arrLen) * unsafe.Sizeof(arr[0]))

	var err error
	start = time.Now()
	devArr, err := cuda.DeviceMemAlloc(arrByteSize)
	took = time.Since(start)
	panicErr(err)
	defer devArr.Free()
	report("DevMallocNoReg", took)

	start = time.Now()
	err = devArr.MemcpyToDevice(unsafe.Pointer(&arr[0]), arrByteSize)
	took = time.Since(start)
	panicErr(err)
	report("MemcpyToDeviceNoReg", took)

	start = time.Now()
	err = devArr.MemcpyFromDevice(unsafe.Pointer(&arr[0]), arrByteSize)
	took = time.Since(start)
	panicErr(err)
	report("MemcpyFromDeviceNoReg", took)
}

func testYesReg(arrLen int) {
	var err error
	//arrByteSize := uint64(uintptr(arrLen) * unsafe.Sizeof(int(0)))
	start := time.Now()
	arr, err := cuda.HostMemAllocWithFlags[int](uint64(arrLen), uint64(unsafe.Sizeof(int(0))), cuda.CU_MEMHOSTALLOC_PORTABLE)
	took := time.Since(start)
	panicErr(err)
	defer arr.Free()
	report("HostMallocCUDA", took)

	for i := 0; i < arrLen; i++ {
		arr.Arr[i] = i
	}

	start = time.Now()
	devArr, err := cuda.DeviceMemAlloc(arr.ActualSize)
	took = time.Since(start)
	panicErr(err)
	defer devArr.Free()
	log("DevMalloc alloced", arr.ActualSize, "Correct?", arr.ActualSize == uint64(uintptr(arrLen)*unsafe.Sizeof(int(0))))
	report("DevMallocYesReg", took)

	start = time.Now()
	err = devArr.MemcpyToDevice(unsafe.Pointer(&arr.Arr[0]), arr.ActualSize)
	took = time.Since(start)
	panicErr(err)
	report("MemcpyToDeviceYesReg", took)

	start = time.Now()
	err = devArr.MemcpyFromDevice(unsafe.Pointer(&arr.Arr[0]), arr.ActualSize)
	took = time.Since(start)
	panicErr(err)
	report("MemcpyFromDeviceYesReg", took)
}

type generatorParam struct {
	name  string
	times int
	fun   func()
}

type generatorState struct {
	leftToRun map[string]int
	allToRun  int

	nextTest int
	numTests int
}

// Outputs a generator function that randomly selects a test to run
func generator(frame *generatorState, tests ...generatorParam) func() func() {
	frame.leftToRun = make(map[string]int, len(tests))
	frame.allToRun = 0
	frame.nextTest = 0
	frame.numTests = len(tests)

	for _, test := range tests {
		frame.leftToRun[test.name] = test.times
		frame.allToRun += test.times
	}

	return func() func() {
		//round robin
		for {
			if frame.allToRun == 0 {
				return nil
			}
			test := tests[frame.nextTest]
			if frame.leftToRun[test.name] > 0 {
				frame.leftToRun[test.name]--
				frame.allToRun--
				frame.nextTest = (frame.nextTest + 1) % frame.numTests
				return test.fun
			}
			frame.nextTest = (frame.nextTest + 1) % frame.numTests
		}
	}
}

func logf(format string, parmas ...any) {
	fmt.Fprintf(os.Stderr, format, parmas...)
}

func log(msg ...any) {
	fmt.Fprintln(os.Stderr, msg...)
}

func panicErr(err error) {
	if err != nil {
		panic(err)
	}
}
