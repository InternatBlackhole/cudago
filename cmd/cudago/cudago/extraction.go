package main

import (
	"errors"
	"io"
	"regexp"
	"strings"
	"unicode/utf8"
)

const (
	// kernelRegex is the regex to match a CUDA kernel function
	// format: __global__ void kernelName(type1 arg1, type2 arg2, ...)
	kernelRegex = `__global__\s+void\s+(\w+)\s*\(([^)]*)\)`
	// constRegex is the regex to match a CUDA constant
	// format: __constant__ type constName
	constRegex = `__constant__\s+([^\(\)]*);`
	// varRegex is the regex to match a CUDA variable
	// format: __device__ type varName
	varRegex = `__device__\s+([^\(\)]*);`
)

var (
	kernelRegexCompiled = regexp.MustCompile(kernelRegex)
	constRegexCompiled  = regexp.MustCompile(constRegex)
	varRegexCompiled    = regexp.MustCompile(varRegex)
)

type RuneReader struct {
	io.Reader
	buf     []byte
	bufLoc  int
	fileLoc int
}

func NewRuneReader(r io.Reader) *RuneReader {
	return &RuneReader{
		Reader:  r,
		buf:     make([]byte, 4096),
		bufLoc:  -1,
		fileLoc: 0,
	}
}

// ReadRune reads a single UTF-8 encoded Unicode character and its size in bytes from the RuneReader.
func (r *RuneReader) ReadRune() (rune, int, error) {
	if r.bufLoc == -1 || r.bufLoc >= len(r.buf) {
		n, err := r.Reader.Read(r.buf)
		if err != nil {
			return 0, 0, err
		}
		r.fileLoc += n
		r.bufLoc = 0
	}
	//ughhh, the buffer may not caintain the full rune, fix
	rn, size := utf8.DecodeRune(r.buf[r.bufLoc:])

	//this is an edge case, will only happen if a rune is split across two buffers, at end of buffer
	if rn == utf8.RuneError && size == 1 {
		//invalid rune, possibly not enough bytes, read the first rune byte and read extra bytes
		first := r.buf[r.bufLoc]
		var n, arrSize int
		var err error
		var arr []byte

		if first <= 0b01111111 { //this is ascii, should never happen
			panic("ascii rune, should never be here")
		} else if first >= 0b11000000 && first <= 0b11011111 { //2 byte rune
			//read the next byte
			arrSize = 2
		} else if /*first >= 0b11100000 &&*/ first <= 0b11101111 { //3 byte rune
			//read the next two bytes
			arrSize = 3
		} else if /*first >= 0b11110000 &&*/ first <= 0b11110111 { //4 byte rune
			//read the next three bytes
			arrSize = 4
		} else {
			return 0, 0, errors.New("invalid utf8 rune")
		}
		arr = make([]byte, arrSize)
		arr[0] = first
		n, err = r.Reader.Read(arr[1:])
		if err != nil {
			return 0, 0, err
		}
		rn, size = utf8.DecodeRune(arr)
		if rn == utf8.RuneError && size == 1 {
			panic("invalid utf8 rune decode after trying to recover")
		}
		r.fileLoc += n
	}
	r.bufLoc += size
	return rn, size, nil
}

// returns the name and arguments of a kernel
func getKernelNameAndArgs(kernel string, receiver func(name string, args []string) bool) error {
	kernelName := kernelRegexCompiled.FindAllStringSubmatchIndex(kernel, -1)

	if kernelName == nil {
		return errors.New("no match")
	}

	for _, kernelNow := range kernelName {
		//kernel is a slice of matches, formula [2*n:2*n+2] is the start and end of the nth match, first match is the full match
		nameLoc := kernelNow[2:4]
		argsLoc := kernelNow[4:6]
		name := kernel[nameLoc[0]:nameLoc[1]]
		args := kernel[argsLoc[0]:argsLoc[1]]
		var splitArgs []string = nil
		if len(args) > 0 {
			splitArgs = strings.Split(args, ",")
			for i, arg := range splitArgs {
				splitArgs[i] = strings.TrimSpace(arg)
			}
		}
		if !receiver(name, splitArgs) {
			break
		}
	}

	return nil
}

// returns the name and type of a constant
func getConstNameAndType(constant string) (string, string, error) {
	constantName := constRegexCompiled.FindStringSubmatch(constant)

	if constantName == nil {
		return "", "", errors.New("no match")
	}

	return extractNameAndType(constantName[1])
}

// returns the name and type of a variable
func getVarNameAndType(variable string) (string, string, error) {
	varName := varRegexCompiled.FindStringSubmatch(variable)

	if varName == nil {
		return "", "", errors.New("no match")
	}

	return extractNameAndType(varName[1])
}

func extractNameAndType(line string) (string, string, error) {
	vars := strings.Fields(line)
	if len(vars) <= 1 {
		return "", "", errors.New("no match")
	}

	name := vars[len(vars)-1]

	typeBuilder := strings.Builder{}
	pointerCount := 0
	lastIsStar := false
	// len - 2 is the last type
	for i := 0; i < len(vars)-1; i++ {
		//current := strings.Trim(vars[i], " ")
		current := vars[i]
		if i == len(vars)-2 {
			pointerCount += strings.Count(current, "*")
			typeBuilder.WriteString(strings.Trim(current, "*"))
			lastIsStar = strings.HasPrefix(current, "*")
		} else if current[0] == '*' {
			pointerCount++
		} else {
			typeBuilder.WriteString(current)
			typeBuilder.WriteString(" ")
		}
	}

	pointerCount += strings.Count(name, "*")
	name = strings.Trim(name, "* ")

	if pointerCount > 0 && !lastIsStar {
		typeBuilder.WriteString(" ")
	}

	for pointerCount > 0 {
		typeBuilder.WriteString("*")
		pointerCount--
	}

	return name, strings.TrimRight(typeBuilder.String(), " "), nil
}
