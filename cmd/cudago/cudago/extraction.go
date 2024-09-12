package main

import (
	"errors"
	"regexp"
	"strings"
)

const (
	// kernelRegex is the regex to match a CUDA kernel function
	// format: __global__ void kernelName(type1 arg1, type2 arg2, ...)
	kernelRegex = `(?:extern\s+"C")?\s*__global__\s+void\s+([[:alpha:]_]\w*)\s*\(([^)]*)\)\s*;*`

	// constRegex is the regex to match a CUDA constant
	//constRegex = `__constant__\s+([^\(\)]*);`
	// varRegex is the regex to match a CUDA variable
	//varRegex = `__device__\s+([^\(\)]*);`

	// regexConstVar is the regex to match a CUDA constant or device variable
	// format: __device__ type varName
	// format: __constant__ type constName
	regexConstVar = `(?:extern\s+"C")?\s*__(device|constant)__\s+([^\(\);]*);+\s*`
)

type definedLocationType int

const (
	TYPE_DEVICE_CONST definedLocationType = iota
	TYPE_DEVICE_VAR   definedLocationType = iota
	TYPE_KERNEL_FUNC  definedLocationType = iota
	TYPE_DEVICE_FUNC  definedLocationType = iota
)

func keywordToType(keyword string, isFunc bool) definedLocationType {
	switch keyword {
	case "device":
		if isFunc {
			return TYPE_DEVICE_FUNC
		}
		return TYPE_DEVICE_VAR
	case "constant":
		return TYPE_DEVICE_CONST
	case "global":
		return TYPE_KERNEL_FUNC
	default:
		return -1
	}
}

var (
	kernelRegexCompiled = regexp.MustCompile(kernelRegex)
	//constRegexCompiled    = regexp.MustCompile(constRegex)
	//varRegexCompiled      = regexp.MustCompile(varRegex)
	regexConstVarCompiled = regexp.MustCompile(regexConstVar)
)

// returns the name and arguments of a kernel
func getKernelNameAndArgs(kernel string, receiver func(name string, args []string) bool) error {
	//TODO: could be merged with processInput, look into named capture groups
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

func getDefinedVariables(input string, receiver func(name string, ctyp string, typ definedLocationType) bool) error {
	matches := regexConstVarCompiled.FindAllStringSubmatchIndex(input, -1)

	if matches == nil {
		return errors.New("no match")
	}

	//match can be __device__ or __constant__
	for _, match := range matches {
		nameAndTypeLoc := match[4:6]
		actual := input[nameAndTypeLoc[0]:nameAndTypeLoc[1]]
		name, ctyp, err := extractNameAndType(actual)
		if err != nil {
			return err
		}

		definedLoc := match[2:4]
		defined := input[definedLoc[0]:definedLoc[1]]
		typ := keywordToType(defined, false)
		if !receiver(name, ctyp, typ) {
			break
		}
	}

	return nil
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
