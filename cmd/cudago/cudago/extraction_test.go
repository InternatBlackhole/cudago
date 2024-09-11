package main

import (
	"io"
	"reflect"
	"strings"
	"testing"
)

func Test_getDefinedVariables(t *testing.T) {
	type args struct {
		input string
	}
	tests := []struct {
		name      string
		args      args
		wantName  []string
		wantCType []string
		wantType  []definedLocationType
		wantErr   bool
	}{
		{
			name:      "unsigned char constant",
			args:      args{"__constant__ unsigned char varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},
		{
			name:      "int constant",
			args:      args{"__constant__ int varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"int"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},
		{
			name:      "array constant",
			args:      args{"__constant__ int arrName[10];"},
			wantName:  []string{"arrName[10]"},
			wantCType: []string{"int"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},
		{
			name:      "long long constant",
			args:      args{"__constant__ long long varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"long long"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},
		{
			name:      "unsigned long long constant",
			args:      args{"__constant__ unsigned long long varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned long long"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},
		{
			name:      "pointer at var name constant",
			args:      args{"__constant__ unsigned char *varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char *"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},
		{
			name:      "pointer at type constant",
			args:      args{"__constant__ unsigned char* varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char *"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},
		{
			name:      "pointer in middle constant",
			args:      args{"__constant__ unsigned char * varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char *"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},
		{
			name:      "double pointer constant",
			args:      args{"__constant__ unsigned char ** varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char **"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},
		{
			name:      "double with space constant",
			args:      args{"__constant__ unsigned char * * varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char **"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},
		{
			name:      "triple with space constant",
			args:      args{"__constant__ unsigned char * * * varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char ***"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},
		{
			name:      "line break constant",
			args:      args{"__constant__ unsigned char\n* varName;\n"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char *"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST},
			wantErr:   false,
		},

		{
			name:      "unsigned char device",
			args:      args{"__device__ unsigned char varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char"},
			wantType:  []definedLocationType{TYPE_DEVICE_VAR},
			wantErr:   false,
		},
		{
			name:      "int device",
			args:      args{"__device__ int varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"int"},
			wantType:  []definedLocationType{TYPE_DEVICE_VAR},
			wantErr:   false,
		},
		{
			name:      "array device",
			args:      args{"__device__ int arrName[10];"},
			wantName:  []string{"arrName[10]"},
			wantCType: []string{"int"},
			wantType:  []definedLocationType{TYPE_DEVICE_VAR},
			wantErr:   false,
		},
		{
			name:      "long long device",
			args:      args{"__device__ long long varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"long long"},
			wantType:  []definedLocationType{TYPE_DEVICE_VAR},
			wantErr:   false,
		},
		{
			name:      "unsigned long long device",
			args:      args{"__device__ unsigned long long varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned long long"},
			wantType:  []definedLocationType{TYPE_DEVICE_VAR},
			wantErr:   false,
		},
		{
			name:      "pointer at var name device",
			args:      args{"__device__ unsigned char *varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char *"},
			wantType:  []definedLocationType{TYPE_DEVICE_VAR},
			wantErr:   false,
		},
		{
			name:      "pointer at type device",
			args:      args{"__device__ unsigned char* varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char *"},
			wantType:  []definedLocationType{TYPE_DEVICE_VAR},
			wantErr:   false,
		},
		{
			name:      "pointer in middle device",
			args:      args{"__device__ unsigned char * varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char *"},
			wantType:  []definedLocationType{TYPE_DEVICE_VAR},
			wantErr:   false,
		},
		{
			name:      "double pointer device",
			args:      args{"__device__ unsigned char ** varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char **"},
			wantType:  []definedLocationType{TYPE_DEVICE_VAR},
			wantErr:   false,
		},
		{
			name:      "double with space device",
			args:      args{"__device__ unsigned char * * varName;"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char **"},
			wantType:  []definedLocationType{TYPE_DEVICE_VAR},
			wantErr:   false,
		},
		{
			name:      "line break device",
			args:      args{"__device__ unsigned char\n* varName;\n"},
			wantName:  []string{"varName"},
			wantCType: []string{"unsigned char *"},
			wantType:  []definedLocationType{TYPE_DEVICE_VAR},
			wantErr:   false,
		},

		{
			name:      "unsigned char device and constant",
			args:      args{"__constant__ unsigned\t long* varName1;\n__device__ unsigned\n char varName2;"},
			wantName:  []string{"varName1", "varName2"},
			wantCType: []string{"unsigned long *", "unsigned char"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST, TYPE_DEVICE_VAR},
			wantErr:   false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotNameCorr, gotCTypeCorr, gotTypeCorr, index := false, false, false, -1
			gotNameLatest, gotCTypeLatest, gotTypeLatest := "", "", definedLocationType(0)
			err := getDefinedVariables(tt.args.input,
				func(name string, ctyp string, typ definedLocationType) bool {
					index++
					gotNameCorr = name == tt.wantName[index]
					gotCTypeCorr = ctyp == tt.wantCType[index]
					gotTypeCorr = typ == tt.wantType[index]
					gotNameLatest = name
					gotCTypeLatest = ctyp
					gotTypeLatest = typ
					//continue if all are correct
					return gotNameCorr && gotCTypeCorr && gotTypeCorr
				})
			if err != nil {
				//error occured
				if !tt.wantErr {
					t.Errorf("getDefinedVariables() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}
			if !gotNameCorr {
				t.Errorf("getDefinedVariables() gotName = %v, want %v", gotNameLatest, tt.wantName[index])
			}
			if !gotCTypeCorr {
				t.Errorf("getDefinedVariables() gotCType = %v, want %v", gotCTypeLatest, tt.wantCType[index])
			}
			if !gotTypeCorr {
				t.Errorf("getDefinedVariables() gotType = %v, want %v", gotTypeLatest, tt.wantType[index])
			}
		})
	}
}

func Test_getKernelNameAndArgs(t *testing.T) {
	type args struct {
		kernel string
	}
	tests := []struct {
		name    string
		args    args
		want    []string
		want1   [][]string
		wantErr bool
	}{
		{
			name:    "ptrs and ints",
			args:    args{`extern "C" __global__ void borders(unsigned char *origImage, int width, int height, unsigned char *gradient, int imgSize)`},
			want:    []string{"borders"},
			want1:   [][]string{{"unsigned char *origImage", "int width", "int height", "unsigned char *gradient", "int imgSize"}},
			wantErr: false,
		},
		{
			name:    "arrParams",
			args:    args{"__global__ void params(float A[N][N], float B[N][N], float C[N][N], float alpha, float beta, float **params)"},
			want:    []string{"params"},
			want1:   [][]string{{"float A[N][N]", "float B[N][N]", "float C[N][N]", "float alpha", "float beta", "float **params"}},
			wantErr: false,
		},
		{
			name:    "noArgs",
			args:    args{"__global__ void noArgs()"},
			want:    []string{"noArgs"},
			want1:   [][]string{nil},
			wantErr: false,
		},
		{
			name:    "noName",
			args:    args{"__global__ void ()"},
			want:    []string{""},
			want1:   [][]string{nil},
			wantErr: true,
		},
		{
			name:    "unsigned long long args",
			args:    args{"__global__ void uborders(unsigned long long *origImage, unsigned long long width, unsigned long long height, unsigned long long *gradient, unsigned long long imgSize)"},
			want:    []string{"uborders"},
			want1:   [][]string{{"unsigned long long *origImage", "unsigned long long width", "unsigned long long height", "unsigned long long *gradient", "unsigned long long imgSize"}},
			wantErr: false,
		},
		{
			name:    "line break",
			args:    args{"__global__ \nvoid\nborders\n(unsigned char *origImage,\n int width, int height,\n unsigned char *gradient, int imgSize)"},
			want:    []string{"borders"},
			want1:   [][]string{{"unsigned char *origImage", "int width", "int height", "unsigned char *gradient", "int imgSize"}},
			wantErr: false,
		},
		{
			name:    "multiple kernels",
			args:    args{"\n  \t__global__ void kernel1(int* a, int b)\n\n    __global__    void kernel2(unsigned int c, int d)"},
			want:    []string{"kernel1", "kernel2"},
			want1:   [][]string{{"int* a", "int b"}, {"unsigned int c", "int d"}},
			wantErr: false,
		},
		{
			name:    "only header",
			args:    args{"__global__ void \nkernel1(); "},
			want:    []string{"kernel1"},
			want1:   [][]string{nil},
			wantErr: false,
		},
		{
			name:    "with body",
			args:    args{"__global__ void kernel1() { int a = 0; }"},
			want:    []string{"kernel1"},
			want1:   [][]string{nil},
			wantErr: false,
		},
		{
			name:    "with body and args",
			args:    args{"__global__ void kernel1(int a) { int b = 0; }"},
			want:    []string{"kernel1"},
			want1:   [][]string{{"int a"}},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotNameCorrect, gotArgsCorrect, index := false, false, 0
			gotNameLatest := ""
			var gotArgsLatest []string = nil
			err := getKernelNameAndArgs(tt.args.kernel,
				func(name string, args []string) bool {
					gotNameCorrect = name == tt.want[index]
					gotArgsCorrect = reflect.DeepEqual(args, tt.want1[index])
					gotNameLatest = name
					gotArgsLatest = args
					index++
					return gotNameCorrect && gotArgsCorrect // continue if both are correct
				})
			if err != nil {
				//error occured
				if !tt.wantErr {
					t.Errorf("getKernelNameAndArgs() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}
			if !gotNameCorrect {
				t.Errorf("getKernelNameAndArgs() gotName = %v, want %v", gotNameLatest, tt.want[index])
			}
			if !gotArgsCorrect {
				t.Errorf("getKernelNameAndArgs() gotArgs = %v, want %v", gotArgsLatest, tt.want1[index])
			}
		})
	}
}

func Test_extractNameAndType(t *testing.T) {
	type args struct {
		line string
	}
	tests := []struct {
		name    string
		args    args
		want    string
		want1   string
		wantErr bool
	}{
		{
			name:    "unsigned char",
			args:    args{"unsigned char varName"},
			want:    "varName",
			want1:   "unsigned char",
			wantErr: false,
		},
		{
			name:    "int",
			args:    args{"int varName"},
			want:    "varName",
			want1:   "int",
			wantErr: false,
		},
		{
			name:    "array",
			args:    args{"int arrName[10]"},
			want:    "arrName[10]",
			want1:   "int",
			wantErr: false,
		},
		{
			name:    "long long",
			args:    args{"long long varName"},
			want:    "varName",
			want1:   "long long",
			wantErr: false,
		},
		{
			name:    "unsigned long long",
			args:    args{"unsigned long long varName"},
			want:    "varName",
			want1:   "unsigned long long",
			wantErr: false,
		},
		{
			name:    "pointer at var name",
			args:    args{"unsigned char *varName"},
			want:    "varName",
			want1:   "unsigned char *",
			wantErr: false,
		},
		{
			name:    "pointer at type",
			args:    args{"unsigned char* varName"},
			want:    "varName",
			want1:   "unsigned char *",
			wantErr: false,
		},
		{
			name:    "pointer in middle",
			args:    args{"unsigned char * varName"},
			want:    "varName",
			want1:   "unsigned char *",
			wantErr: false,
		},
		{
			name:    "double pointer",
			args:    args{"unsigned char ** varName"},
			want:    "varName",
			want1:   "unsigned char **",
			wantErr: false,
		},
		{
			name:    "double with space",
			args:    args{"unsigned char * * varName"},
			want:    "varName",
			want1:   "unsigned char **",
			wantErr: false,
		},
		{
			name:    "triple with space",
			args:    args{"unsigned char * * * varName"},
			want:    "varName",
			want1:   "unsigned char ***",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := extractNameAndType(tt.args.line)
			if (err != nil) != tt.wantErr {
				t.Errorf("extractNameAndType() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("extractNameAndType() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("extractNameAndType() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

/*func Test_getKernelNameAndArgsReader(t *testing.T) {
	type args struct {
		reader io.Reader
	}
	tests := []struct {
		name    string
		args    args
		want    string
		want1   []string
		wantErr bool
	}{
		{
			name:    "ptrs and ints",
			args:    args{strings.NewReader(`extern "C" __global__ void ćborders(unsigned char *origImage, int widthđ, int žheight, unsigned char *gradient, int imgSize)`)},
			want:    "ćborders",
			want1:   []string{"unsigned char *origImage", "int widthđ", "int žheight", "unsigned char *gradient", "int imgSize"},
			wantErr: false,
		},
		{
			name:    "arrParams",
			args:    args{strings.NewReader("__global__ void paramš(float A[N][N], float B[N][N], float C[N][N], float alpha, float ßbeta, float **params)")},
			want:    "paramš",
			want1:   []string{"float A[N][N]", "float B[N][N]", "float C[N][N]", "float alpha", "float ßbeta", "float **params"},
			wantErr: false,
		},
		{
			name:    "noArgs",
			args:    args{strings.NewReader("__global__ void noArgš()")},
			want:    "noArgš",
			want1:   nil,
			wantErr: false,
		},
		{
			name:    "noName",
			args:    args{strings.NewReader("__global__ void ()")},
			want:    "",
			want1:   nil,
			wantErr: true,
		},
		{
			name:    "unsigned long long args",
			args:    args{strings.NewReader("__global__ void uborders(unsigned lo÷ng long *origImage, unsigned long long width, unsigned long long height, unsigned long long *gradient, unsigned long long imgSize)")},
			want:    "uborders",
			want1:   []string{"unsigned lo÷ng long *origImage", "unsigned long long width", "unsigned long long height", "unsigned long long *gradient", "unsigned long long imgSize"},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := getKernelNameAndArgsReader(tt.args.reader)
			if (err != nil) != tt.wantErr {
				t.Errorf("getKernelNameAndArgsReader() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("getKernelNameAndArgsReader() got = %v, want %v", got, tt.want)
			}
			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("getKernelNameAndArgsReader() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}*/

func TestRuneReader_ReadRune(t *testing.T) {
	type fields struct {
		Reader  io.Reader
		buf     []byte
		bufLoc  int
		fileLoc int
	}
	tests := []struct {
		name    string
		fields  fields
		want    rune
		want1   int
		wantErr bool
	}{
		{
			name: "1 byte rune",
			fields: fields{
				Reader: strings.NewReader("a"),
				buf:    make([]byte, 1),
				bufLoc: -1,
			},
			want:    'a',
			want1:   1,
			wantErr: false,
		},
		{
			name: "2 byte rune",
			fields: fields{
				Reader: strings.NewReader("Ć"),
				buf:    make([]byte, 2),
				bufLoc: -1,
			},
			want:    'Ć',
			want1:   2,
			wantErr: false,
		},
		{
			name: "3 byte rune",
			fields: fields{
				Reader: strings.NewReader("₿"),
				buf:    make([]byte, 3),
				bufLoc: -1,
			},
			want:    '₿',
			want1:   3,
			wantErr: false,
		},
		{
			name: "4 byte rune",
			fields: fields{
				Reader: strings.NewReader("𐲕"),
				buf:    make([]byte, 4),
				bufLoc: -1,
			},
			want:    '𐲕',
			want1:   4,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &RuneReader{
				Reader:  tt.fields.Reader,
				buf:     tt.fields.buf,
				bufLoc:  tt.fields.bufLoc,
				fileLoc: tt.fields.fileLoc,
			}
			got, got1, err := r.ReadRune()
			if (err != nil) != tt.wantErr {
				t.Errorf("RuneReader.ReadRune() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("RuneReader.ReadRune() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("RuneReader.ReadRune() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}
