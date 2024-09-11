package main

import (
	"io"
	"reflect"
	"strings"
	"testing"
)

func Test_getConstNameAndType(t *testing.T) {
	type args struct {
		constant string
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
			args:    args{"__constant__ unsigned char varName;"},
			want:    "varName",
			want1:   "unsigned char",
			wantErr: false,
		},
		{
			name:    "int",
			args:    args{"__constant__ int varName;"},
			want:    "varName",
			want1:   "int",
			wantErr: false,
		},
		{
			name:    "array",
			args:    args{"__constant__ int arrName[10];"},
			want:    "arrName[10]",
			want1:   "int",
			wantErr: false,
		},
		{
			name:    "long long",
			args:    args{"__constant__ long long varName;"},
			want:    "varName",
			want1:   "long long",
			wantErr: false,
		},
		{
			name:    "unsigned long long",
			args:    args{"__constant__ unsigned long long varName;"},
			want:    "varName",
			want1:   "unsigned long long",
			wantErr: false,
		},
		{
			name:    "pointer at var name",
			args:    args{"__constant__ unsigned char *varName;"},
			want:    "varName",
			want1:   "unsigned char *",
			wantErr: false,
		},
		{
			name:    "pointer at type",
			args:    args{"__constant__ unsigned char* varName;"},
			want:    "varName",
			want1:   "unsigned char *",
			wantErr: false,
		},
		{
			name:    "pointer in middle",
			args:    args{"__constant__ unsigned char * varName;"},
			want:    "varName",
			want1:   "unsigned char *",
			wantErr: false,
		},
		{
			name:    "double pointer",
			args:    args{"__constant__ unsigned char ** varName;"},
			want:    "varName",
			want1:   "unsigned char **",
			wantErr: false,
		},
		{
			name:    "double with space",
			args:    args{"__constant__ unsigned char * * varName;"},
			want:    "varName",
			want1:   "unsigned char **",
			wantErr: false,
		},
		{
			name:    "triple with space",
			args:    args{"__constant__ unsigned char * * * varName;"},
			want:    "varName",
			want1:   "unsigned char ***",
			wantErr: false,
		},
		{
			name:    "line break",
			args:    args{"__constant__ unsigned char\n* varName;\n"},
			want:    "varName",
			want1:   "unsigned char *",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := getConstNameAndType(tt.args.constant)
			if (err != nil) != tt.wantErr {
				t.Errorf("getConstNameAndType() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("getConstNameAndType() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("getConstNameAndType() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_getVarNameAndType(t *testing.T) {
	type args struct {
		variable string
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
			args:    args{"__device__ unsigned char varName;"},
			want:    "varName",
			want1:   "unsigned char",
			wantErr: false,
		},
		{
			name:    "int",
			args:    args{"__device__ int varName;"},
			want:    "varName",
			want1:   "int",
			wantErr: false,
		},
		{
			name:    "array",
			args:    args{"__device__ int arrName[10];"},
			want:    "arrName[10]",
			want1:   "int",
			wantErr: false,
		},
		{
			name:    "long long",
			args:    args{"__device__ long long varName;"},
			want:    "varName",
			want1:   "long long",
			wantErr: false,
		},
		{
			name:    "unsigned long long",
			args:    args{"__device__ unsigned long long varName;"},
			want:    "varName",
			want1:   "unsigned long long",
			wantErr: false,
		},
		{
			name:    "pointer at var name",
			args:    args{"__device__ unsigned char *varName;"},
			want:    "varName",
			want1:   "unsigned char *",
			wantErr: false,
		},
		{
			name:    "pointer at type",
			args:    args{"__device__ unsigned char* varName;"},
			want:    "varName",
			want1:   "unsigned char *",
			wantErr: false,
		},
		{
			name:    "pointer in middle",
			args:    args{"__device__ unsigned char * varName;"},
			want:    "varName",
			want1:   "unsigned char *",
			wantErr: false,
		},
		{
			name:    "double pointer",
			args:    args{"__device__ unsigned char ** varName;"},
			want:    "varName",
			want1:   "unsigned char **",
			wantErr: false,
		},
		{
			name:    "double with space",
			args:    args{"__device__ unsigned char * * varName;"},
			want:    "varName",
			want1:   "unsigned char **",
			wantErr: false,
		},
		{
			name:    "line break",
			args:    args{"__device__ unsigned char\n* varName;\n"},
			want:    "varName",
			want1:   "unsigned char *",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := getVarNameAndType(tt.args.variable)
			if (err != nil) != tt.wantErr {
				t.Errorf("getVarNameAndType() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("getVarNameAndType() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("getVarNameAndType() got1 = %v, want %v", got1, tt.want1)
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
			args:    args{"\n  \t__global__ void kernel1(int* a, int b)\n\n    __global__    void kernel2(int c, int d)"},
			want:    []string{"kernel1", "kernel2"},
			want1:   [][]string{{"int* a", "int b"}, {"int c", "int d"}},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotCorrect, got1Correct, index := false, false, 0
			gotLatest := ""
			var got1Latest []string = nil
			err := getKernelNameAndArgs(tt.args.kernel,
				func(got string, got1 []string) bool {
					gotCorrect = got == tt.want[index]
					gotLatest = got
					if !gotCorrect {
						index++
						return false
					}
					got1Correct = reflect.DeepEqual(got1, tt.want1[index])
					got1Latest = got1
					index++
					return got1Correct // continue if both are correct
				})
			if err != nil {
				//error occured
				if !tt.wantErr {
					t.Errorf("getKernelNameAndArgs() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}
			if !gotCorrect {
				t.Errorf("getKernelNameAndArgs() got = %v, want %v", gotLatest, tt.want[index])
			}
			if !got1Correct {
				t.Errorf("getKernelNameAndArgs() got1 = %v, want %v", got1Latest, tt.want1[index])
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
