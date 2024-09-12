package main

import (
	"reflect"
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
		{
			name:      "ultimate test",
			args:      args{cudaFileCuTest},
			wantName:  []string{"randomConst", "randomLongVar", "random_Const"},
			wantCType: []string{"int", "unsigned long long long *", "uint16_t"},
			wantType:  []definedLocationType{TYPE_DEVICE_CONST, TYPE_DEVICE_VAR, TYPE_DEVICE_CONST},
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
		{
			name: "ultimate test",
			args: args{cudaFileCuTest},
			want: []string{"bitonicSortStart", "bitonicSortMiddle", "bitonicSortFinish"},
			want1: [][]string{{"int *a", "int len"}, {"int *a", "int len", "int k", "int j"},
				{"int *a", "int len", "int k"}},
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

const cudaFileCuTest = `
// bitonično urejanje tabele celih števil
// 		argumenta: število niti v bloku in velikost tabele
//		elementi tabele so inicializirani naključno
// s sinhornizacijo niti v bloku se v največji možni meri izognemo globalni sinhornizaciji
// bitonicSort je zdaj funkcija na napravi, ki jo kličejo trije ščepci
// bitonicSortStart in bitonicSortFinish urejata v skupnem pomnilniku

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda.h"
#include "helper_cuda.h"

__device__ void bitonicSort(int *a, int len, int k, int j) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;    
    if (gid < len/2) {
		int i1 = 2*j * (int)(gid / j) + (gid % j);	// prvi element
		int i2 = i1 ^ j;							// drugi element
		int dec = i1 & k;							// smer urejanja (padajoče: dec != 0)
		if ((dec == 0 && a[i1] > a[i2]) || (dec != 0 && a[i1] < a[i2])) {
			int temp = a[i1];
			a[i1] = a[i2];
			a[i2] = temp;
		}
	}
}

__device__ void bitonicSortShared(int *as, int len, int k, int j) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;    
    if (gid < len/2) {
		int i1 = 2*j * (int)(gid / j) + (gid % j);	// prvi element
		int i2 = i1 ^ j;							// drugi element
		int dec = i1 & k;							// smer urejanja (padajoče: dec != 0)
		int i1s = i1 % blockDim.x;
		int i2s = i2 % blockDim.x;
		if ((dec == 0 && as[i1s] > as[i2s]) || (dec != 0 && as[i1s] < as[i2s])) {
			int temp = as[i1s];
			as[i1s] = as[i2s];
			as[i2s] = temp;
		}
	}
}

__device__ void copyToShared(int *as, int *a) {
	int i1Start = 2 * blockDim.x * blockIdx.x;
	as[threadIdx.x] = a[i1Start + threadIdx.x];	
	as[blockDim.x + threadIdx.x] = a[i1Start + blockDim.x + threadIdx.x];
}

__device__ void copyFromShared(int *a, int *as) {int *a, int len, int k, int j
	int i1Start = 2 * blockDim.x * blockIdx.x;
	a[i1Start + threadIdx.x] = as[threadIdx.x];	
	a[i1Start + blockDim.x + threadIdx.x] = as[blockDim.x + threadIdx.x];
}

extern "C" __global__ void bitonicSortStart(int *a, int len) {
	extern __shared__ int as[];
	copyToShared(as, a);
	for (int k = 2; k <= 2 * blockDim.x; k <<= 1) 
		for (int j = k/2; j > 0; j >>= 1) {
			bitonicSortShared(as, len, k, j);
			__syncthreads();
	}
	copyFromShared(a, as);
}

extern "C" __global__ void bitonicSortMiddle(int *a, int len, int k, int j) {
	bitonicSort(a, len, k, j);
}

__global__ void bitonicSortFinish(int *a, int len, int k) {
	extern __shared__ int as[];
	copyToShared(as, a);
	for (int j = 2*blockDim.x; j > 0; j >>= 1) {
		bitonicSortShared(as, len, k, j);
		__syncthreads();
	}
	copyFromShared(a, as);
}

__constant__ 
int randomConst;

__device__ unsigned long long
 long * randomLongVar;


__constant__ uint16_t random_Const;


int main(int argc, char **argv) {
	// preberemo argumente iz ukazne vrstice
	int numThreads = 0;
	int tableLength = 0;
	if (argc == 3) {
		numThreads = atoi(argv[1]);
		tableLength = atoi(argv[2]);
	}
	if (numThreads <= 0 || tableLength <= 0 || ceil(log2(tableLength)) != floor(log2(tableLength))) {
		printf("usage:\n\t%s <number of block threads> <table length (power of 2)>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	// določimo potrebno število blokov niti (rabimo toliko niti, kot je parov elemntov)
	int numBlocks = (tableLength/2 - 1) / numThreads + 1;

	// rezerviramo pomnilnik na gostitelju
	int *a = (int *)malloc(tableLength * sizeof(int));
	int *ha = (int *)malloc(tableLength * sizeof(int));
	
	// rezerviramo pomnilnik na napravi
	int *da;
	checkCudaErrors(cudaMalloc((void**)&da, tableLength * sizeof(int)));

	// nastavimo vrednosti tabel a in ha na gostitelju
	srand(time(NULL));
	for (int i = 0; i < tableLength; i++) {
        a[i] = rand();
		ha[i] = a[i];
    }

	// merjenje časa na napravi - začetek
	struct timespec startDevice, stopDevice;
	clock_gettime(CLOCK_MONOTONIC, &startDevice);

	// prenesemo tabelo a iz gostitelja na napravo
	checkCudaErrors(cudaMemcpy(da, ha, tableLength * sizeof(int), cudaMemcpyHostToDevice));

	// zaženemo kodo na napravi
	dim3 gridSize(numBlocks, 1, 1);
	dim3 blockSize(numThreads, 1, 1);

	bitonicSortStart<<<gridSize, blockSize, 2*blockSize.x*sizeof(int)>>>(da, tableLength);			// k = 2 ... 2 * blockSize.x
    for (int k = 4 * blockSize.x; k <= tableLength; k <<= 1) {										// k = 4 * blockSize ... tableLength
        for (int j = k/2; j > 2 * blockSize.x; j >>= 1) {											//   j = k/2 ... 2 * blockSize.x
        	bitonicSortMiddle<<<gridSize, blockSize, 2*blockSize.x*sizeof(int)>>>(da, tableLength, k, j);
	        checkCudaErrors(cudaGetLastError());
        }
		bitonicSortFinish<<<gridSize, blockSize, 2*blockSize.x*sizeof(int)>>>(da, tableLength, k);	//   j = 2 * blockSize.x ... 1
	}

	// počakamo, da vse niti na napravi zaključijo
	checkCudaErrors(cudaDeviceSynchronize());

	// tabelo a prekopiramo iz naprave na gostitelja
	checkCudaErrors(cudaMemcpy(ha, da, tableLength * sizeof(int), cudaMemcpyDeviceToHost));

	// merjenje časa na napravi - konec
	clock_gettime(CLOCK_MONOTONIC, &stopDevice);
	double timeDevice = (stopDevice.tv_sec - startDevice.tv_sec) * 1e3 + (stopDevice.tv_nsec - startDevice.tv_nsec) / 1e6;

	// urejanje na gostitelju
	struct timespec startHost, stopHost;
	clock_gettime(CLOCK_MONOTONIC, &startHost);

    int i2, dec, temp;
    for (int k = 2; k <= tableLength; k <<= 1) 
        for (int j = k/2; j > 0; j >>= 1)
            for (int i1 = 0; i1 < tableLength; i1++) {
                i2 = i1 ^ j;
                dec = i1 & k;
                if (i2 > i1)
                    if ((dec == 0 && a[i1] > a[i2]) || (dec != 0 && a[i1] < a[i2])) {
                        temp = a[i1];
                        a[i1] = a[i2];
                        a[i2] = temp;
                    }
            }

	clock_gettime(CLOCK_MONOTONIC, &stopHost);
	double timeHost = (stopHost.tv_sec - startHost.tv_sec) * 1e3 + (stopHost.tv_nsec - startHost.tv_nsec) / 1e6;

    // preverimo rešitev
    int okDevice = 1;
    int okHost = 1;
    int previousDevice = ha[0];
    int previousHost = a[0];
    for (int i = 1; i < tableLength; i++) {
        okDevice &= (previousDevice <= ha[i]);
        okHost &= (previousHost <= a[i]);
    }
    printf("Device: %s (%lf ms)\n", okDevice ? "correct" : "wrong", timeDevice);
    printf("Host  : %s (%lf ms)\n", okHost ? "correct" : "wrong", timeHost);

	// sprostimo pomnilnik na napravi
	checkCudaErrors(cudaFree(da));

	// sprostimo pomnilnik na gostitelju
	free(a);
	free(ha);

	return 0;
}
`
