package main

import (
	"fmt"
	"os"
	"time"

	"github.com/InternatBlackhole/cudago/cuda"
)

const (
	//form     = "%s;%d;%d\n"
	//formHead = "Func;Time;RunNum"
	form     = "%s;%f\n"
	formHead = "Func;Time"
)

var (
	grid = cuda.Dim3{
		X: uint32(1),
		Y: uint32(1),
		Z: uint32(1),
	}

	block = cuda.Dim3{
		X: uint32(32),
		Y: uint32(32),
		Z: uint32(1),
	}
)

func report(funcName string, time time.Duration) {
	fmt.Printf(form, funcName, float64(time.Nanoseconds()/1000)) //microseconds
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

	callAllOnce(grid, block)
	log("Prerun done")

	state := generatorState{}

	gen := generator(&state, tests(grid, block)...)
	log("Starting tests")
	for fun := gen(); fun != nil; fun = gen() {
		fun()
	}
	log("Tests done")
	return 0
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
