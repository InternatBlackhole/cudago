package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"path"
	"regexp"
	"strings"

	"github.com/InternatBlackhole/cudago/nvrtc"
)

var (
	isProd         = false
	nvrtcFlags     = ""
	packageName    = "" //output directory and package name
	filesToCompile []string
	//templates are in templates.go

	validPackageNameRegex = regexp.MustCompile(`[^[:digit:]][[:alnum:]]*`)
)

func main() {
	os.Exit(mainWithCode())
}

func mainWithCode() int {
	//get flags from the command line; this programs flags are until '--' is encountered
	flag.BoolVar(&isProd, "prod", false, "Set to true if you want to compile with production flags")
	flag.StringVar(&nvrtcFlags, "nvcc", "", "Flags to pass to nvcc/nvrtc")
	flag.StringVar(&packageName, "package", "", "Package name for the generated code and output directory")

	flag.Parse()

	if packageName == "" {
		fmt.Fprintln(os.Stderr, "No package name provided")
		usage()
		return 1
	}

	if !validPackageNameRegex.MatchString(packageName) {
		fmt.Fprintf(os.Stderr, "Invalid package name: %s\n", packageName)
		return 1
	}

	filesToCompile = flag.Args()

	if len(filesToCompile) == 0 {
		fmt.Fprintln(os.Stderr, "No files to compile")
		usage()
		return 1
	}

	err := os.MkdirAll(packageName, os.ModePerm)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output directory: %v\n", err)
		return 1
	}

	for _, file := range filesToCompile {

		srcFile, err := os.Open(file)
		if err != nil {
			panic(err)
		}
		defer srcFile.Close()

		base := strings.SplitN(path.Base(file), ".", 2)[0]
		outFile, err := os.Create(packageName + "/" + base + ".go")
		if err != nil {
			panic(err)
		}
		defer outFile.Close()

		args := templateArgsFromFile(srcFile)
		args.SetFileName(base)

		if isProd {
			err = createProdFile(args, outFile)
		} else {
			err = createDevFile(args, outFile)
		}

		if err != nil {
			panic(err)
		}
	}

	//outFile, err := os.Create(packageName + "/utilities.go")
	//panicErr(err)
	//defer outFile.Close()
	//err = createUtilityFile(&TemplateArgs{Package: packageName}, outFile)
	//panicErr(err)

	return 0
}

func templateArgsFromFile(file *os.File) *TemplateArgs {
	//TODO: look at bufio.Scanner and SplitFunc
	//reader := bufio.NewScanner(src)

	stat, err := file.Stat()
	panicErr(err)
	fileSize := stat.Size()
	src := bufio.NewReader(file)

	buf := make([]byte, fileSize)
	read, err := io.ReadFull(src, buf)
	panicErr(err)
	if read != int(fileSize) {
		panic("read less bytes than expected")
	}

	program, err := nvrtc.CreateProgram(string(buf), file.Name(), nil)
	nvrtcPanic(err, program)
	defer program.Destroy()
	err = program.Compile(strings.Split(nvrtcFlags, ","))
	nvrtcPanic(err, program)

	ptx, err := program.GetPTX()
	nvrtcPanic(err, program)
	kernels := make([]*CuFileFunc, 0)
	vars := make(map[string]string)
	consts := make(map[string]string)

	err = getKernelNameAndArgs(string(buf), func(name string, args []string) (bool, error) {
		k := NewTemplateFunc()
		k.SetName(name)
		k.SetArgs(args)
		k.IsKernel = true
		kernels = append(kernels, k)
		return true, nil //continue
	})
	panicErr(err)

	err = getDefinedVariables(string(buf), func(name, ctyp string, typ definedLocationType) bool {
		switch typ {
		case TYPE_DEVICE_CONST:
			consts[name] = ctyp
		case TYPE_DEVICE_VAR:
			vars[name] = ctyp
		}
		return true
	})
	if err != errNoMatch {
		panic(err)
	}

	return &TemplateArgs{
		Package:   packageName,
		Funcs:     kernels,
		Constants: consts,
		Variables: vars,
		PTXCode:   string(ptx),
	}
}

func usage() {
	//do something
	flag.Usage()
	fmt.Println("Specify files to compile after '--'")
}

func panicErr(err error) {
	if err != nil {
		panic(err)
	}
}

func nvrtcPanic(erro error, program *nvrtc.Program) {
	if erro == nil {
		return
	}
	log, err := program.GetLog()
	if err != nil {
		panic(err)
	}
	fmt.Printf("NVRTC error: %v\nLog: %s", err, log)
	panic(erro)
}
