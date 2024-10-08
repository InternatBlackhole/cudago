// Code generated by cudago. Edit at your own risk.
package cuda_stuff

import (
    "github.com/InternatBlackhole/cudago/cuda"
	"unsafe"
)


const (
	KeyBitonic = "bitonic"
)


type bitonicsortstartArgs struct {
    a uintptr
    len int32

}
type bitonicsortmiddleArgs struct {
    a uintptr
    len int32
    k int32
    j int32

}
type bitonicsortfinishArgs struct {
    a uintptr
    len int32
    k int32

}

/*var (
    bitonicsortstartArgs = bitonicsortstartArgs{}

    bitonicsortmiddleArgs = bitonicsortmiddleArgs{}

    bitonicsortfinishArgs = bitonicsortfinishArgs{}

)*/







func BitonicSortStart(grid, block cuda.Dim3, a uintptr, len int32) error {
	err := autoloadLib_bitonic()
	if err != nil {
		return err
	}
	kern, err := getKernel("bitonic", "bitonicSortStart")
	if err != nil {
		return err
	}
	params := bitonicsortstartArgs{
	    a: a,
	    len: len,
	
	}
	return kern.Launch(grid, block, unsafe.Pointer(&params.a), unsafe.Pointer(&params.len))
}

func BitonicSortStartEx(grid, block cuda.Dim3, sharedMem uint64, stream *cuda.Stream, a uintptr, len int32) error {
	err := autoloadLib_bitonic()
	if err != nil {
		return err
	}
	kern, err := getKernel("bitonic", "bitonicSortStart")
	if err != nil {
		return err
	}
	params := bitonicsortstartArgs{
	    a: a,
	    len: len,
	
	}
	return kern.LaunchEx(grid, block, sharedMem, stream, unsafe.Pointer(&params.a), unsafe.Pointer(&params.len))
}




func BitonicSortMiddle(grid, block cuda.Dim3, a uintptr, len int32, k int32, j int32) error {
	err := autoloadLib_bitonic()
	if err != nil {
		return err
	}
	kern, err := getKernel("bitonic", "bitonicSortMiddle")
	if err != nil {
		return err
	}
	params := bitonicsortmiddleArgs{
	    a: a,
	    len: len,
	    k: k,
	    j: j,
	
	}
	return kern.Launch(grid, block, unsafe.Pointer(&params.a), unsafe.Pointer(&params.len), unsafe.Pointer(&params.k), unsafe.Pointer(&params.j))
}

func BitonicSortMiddleEx(grid, block cuda.Dim3, sharedMem uint64, stream *cuda.Stream, a uintptr, len int32, k int32, j int32) error {
	err := autoloadLib_bitonic()
	if err != nil {
		return err
	}
	kern, err := getKernel("bitonic", "bitonicSortMiddle")
	if err != nil {
		return err
	}
	params := bitonicsortmiddleArgs{
	    a: a,
	    len: len,
	    k: k,
	    j: j,
	
	}
	return kern.LaunchEx(grid, block, sharedMem, stream, unsafe.Pointer(&params.a), unsafe.Pointer(&params.len), unsafe.Pointer(&params.k), unsafe.Pointer(&params.j))
}




func BitonicSortFinish(grid, block cuda.Dim3, a uintptr, len int32, k int32) error {
	err := autoloadLib_bitonic()
	if err != nil {
		return err
	}
	kern, err := getKernel("bitonic", "bitonicSortFinish")
	if err != nil {
		return err
	}
	params := bitonicsortfinishArgs{
	    a: a,
	    len: len,
	    k: k,
	
	}
	return kern.Launch(grid, block, unsafe.Pointer(&params.a), unsafe.Pointer(&params.len), unsafe.Pointer(&params.k))
}

func BitonicSortFinishEx(grid, block cuda.Dim3, sharedMem uint64, stream *cuda.Stream, a uintptr, len int32, k int32) error {
	err := autoloadLib_bitonic()
	if err != nil {
		return err
	}
	kern, err := getKernel("bitonic", "bitonicSortFinish")
	if err != nil {
		return err
	}
	params := bitonicsortfinishArgs{
	    a: a,
	    len: len,
	    k: k,
	
	}
	return kern.LaunchEx(grid, block, sharedMem, stream, unsafe.Pointer(&params.a), unsafe.Pointer(&params.len), unsafe.Pointer(&params.k))
}



var loaded_bitonic = false


func autoloadLib_bitonic() error {
	if loaded_bitonic {
		return nil
	}
	err := InitLibrary([]byte(Bitonic_ptxCode), "bitonic")
	if err != nil {
		return err
	}
	loaded_bitonic = true
	return nil
}

const Bitonic_ptxCode = `//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-34431801
// Cuda compilation tools, release 12.6, V12.6.20
// Based on NVVM 7.0.1
//

.version 8.5
.target sm_52
.address_size 64

	// .globl	bitonicSortStart

.visible .entry bitonicSortStart(
	.param .u64 bitonicSortStart_param_0,
	.param .u32 bitonicSortStart_param_1
)
{
	.reg .pred 	%p<10>;
	.reg .b32 	%r<34>;
	.reg .b64 	%rd<7>;


	ld.param.u64 	%rd4, [bitonicSortStart_param_0];
	ld.param.u32 	%r15, [bitonicSortStart_param_1];
	mov.u32 	%r1, %ntid.x;
	shl.b32 	%r16, %r1, 1;
	setp.eq.s32 	%p1, %r16, 0;
	@%p1 bra 	$L__BB0_13;

	cvta.to.global.u64 	%rd1, %rd4;
	mov.u32 	%r18, %ctaid.x;
	mov.u32 	%r19, %tid.x;
	mad.lo.s32 	%r2, %r18, %r1, %r19;
	shr.u32 	%r20, %r15, 31;
	add.s32 	%r21, %r15, %r20;
	shr.s32 	%r3, %r21, 1;
	mov.u32 	%r31, 2;
	mov.u32 	%r22, %nctaid.x;
	mul.lo.s32 	%r4, %r22, %r1;

$L__BB0_2:
	setp.lt.s32 	%p2, %r31, 2;
	@%p2 bra 	$L__BB0_12;

	shr.u32 	%r32, %r31, 1;

$L__BB0_4:
	setp.ge.s32 	%p3, %r2, %r3;
	@%p3 bra 	$L__BB0_11;

	shl.b32 	%r8, %r32, 1;
	mov.u32 	%r33, %r2;
	bra.uni 	$L__BB0_6;

$L__BB0_8:
	setp.le.s32 	%p6, %r10, %r11;
	@%p6 bra 	$L__BB0_10;
	bra.uni 	$L__BB0_9;

$L__BB0_6:
	div.s32 	%r23, %r33, %r32;
	mul.lo.s32 	%r24, %r23, %r32;
	sub.s32 	%r25, %r33, %r24;
	mad.lo.s32 	%r26, %r8, %r23, %r25;
	xor.b32  	%r27, %r26, %r32;
	and.b32  	%r28, %r26, %r31;
	setp.eq.s32 	%p4, %r28, 0;
	mul.wide.s32 	%rd5, %r26, 4;
	add.s64 	%rd2, %rd1, %rd5;
	ld.global.u32 	%r10, [%rd2];
	mul.wide.s32 	%rd6, %r27, 4;
	add.s64 	%rd3, %rd1, %rd6;
	ld.global.u32 	%r11, [%rd3];
	@%p4 bra 	$L__BB0_8;

	setp.lt.s32 	%p5, %r10, %r11;
	@%p5 bra 	$L__BB0_9;
	bra.uni 	$L__BB0_10;

$L__BB0_9:
	st.global.u32 	[%rd2], %r11;
	st.global.u32 	[%rd3], %r10;

$L__BB0_10:
	add.s32 	%r33, %r33, %r4;
	setp.lt.s32 	%p7, %r33, %r3;
	@%p7 bra 	$L__BB0_6;

$L__BB0_11:
	bar.sync 	0;
	shr.s32 	%r13, %r32, 1;
	setp.gt.s32 	%p8, %r32, 1;
	mov.u32 	%r32, %r13;
	@%p8 bra 	$L__BB0_4;

$L__BB0_12:
	shl.b32 	%r31, %r31, 1;
	setp.le.u32 	%p9, %r31, %r16;
	@%p9 bra 	$L__BB0_2;

$L__BB0_13:
	ret;

}
	// .globl	bitonicSortMiddle
.visible .entry bitonicSortMiddle(
	.param .u64 bitonicSortMiddle_param_0,
	.param .u32 bitonicSortMiddle_param_1,
	.param .u32 bitonicSortMiddle_param_2,
	.param .u32 bitonicSortMiddle_param_3
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<25>;
	.reg .b64 	%rd<7>;


	ld.param.u64 	%rd4, [bitonicSortMiddle_param_0];
	ld.param.u32 	%r10, [bitonicSortMiddle_param_2];
	ld.param.u32 	%r11, [bitonicSortMiddle_param_3];
	mov.u32 	%r1, %ntid.x;
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %tid.x;
	mad.lo.s32 	%r24, %r12, %r1, %r13;
	ld.param.u32 	%r14, [bitonicSortMiddle_param_1];
	shr.u32 	%r15, %r14, 31;
	add.s32 	%r16, %r14, %r15;
	shr.s32 	%r3, %r16, 1;
	setp.ge.s32 	%p1, %r24, %r3;
	@%p1 bra 	$L__BB1_7;

	cvta.to.global.u64 	%rd1, %rd4;
	shl.b32 	%r4, %r11, 1;
	mov.u32 	%r17, %nctaid.x;
	mul.lo.s32 	%r5, %r17, %r1;
	bra.uni 	$L__BB1_2;

$L__BB1_4:
	setp.le.s32 	%p4, %r7, %r8;
	@%p4 bra 	$L__BB1_6;
	bra.uni 	$L__BB1_5;

$L__BB1_2:
	div.s32 	%r18, %r24, %r11;
	mul.lo.s32 	%r19, %r18, %r11;
	sub.s32 	%r20, %r24, %r19;
	mad.lo.s32 	%r21, %r4, %r18, %r20;
	xor.b32  	%r22, %r21, %r11;
	and.b32  	%r23, %r21, %r10;
	setp.eq.s32 	%p2, %r23, 0;
	mul.wide.s32 	%rd5, %r21, 4;
	add.s64 	%rd2, %rd1, %rd5;
	ld.global.u32 	%r7, [%rd2];
	mul.wide.s32 	%rd6, %r22, 4;
	add.s64 	%rd3, %rd1, %rd6;
	ld.global.u32 	%r8, [%rd3];
	@%p2 bra 	$L__BB1_4;

	setp.lt.s32 	%p3, %r7, %r8;
	@%p3 bra 	$L__BB1_5;
	bra.uni 	$L__BB1_6;

$L__BB1_5:
	st.global.u32 	[%rd2], %r8;
	st.global.u32 	[%rd3], %r7;

$L__BB1_6:
	add.s32 	%r24, %r24, %r5;
	setp.lt.s32 	%p5, %r24, %r3;
	@%p5 bra 	$L__BB1_2;

$L__BB1_7:
	ret;

}
	// .globl	bitonicSortFinish
.visible .entry bitonicSortFinish(
	.param .u64 bitonicSortFinish_param_0,
	.param .u32 bitonicSortFinish_param_1,
	.param .u32 bitonicSortFinish_param_2
)
{
	.reg .pred 	%p<8>;
	.reg .b32 	%r<27>;
	.reg .b64 	%rd<7>;


	ld.param.u64 	%rd4, [bitonicSortFinish_param_0];
	ld.param.u32 	%r12, [bitonicSortFinish_param_1];
	ld.param.u32 	%r13, [bitonicSortFinish_param_2];
	mov.u32 	%r25, %ntid.x;
	setp.lt.s32 	%p1, %r25, 1;
	@%p1 bra 	$L__BB2_10;

	cvta.to.global.u64 	%rd1, %rd4;
	mov.u32 	%r14, %ctaid.x;
	mov.u32 	%r15, %tid.x;
	mad.lo.s32 	%r2, %r14, %r25, %r15;
	shr.u32 	%r16, %r12, 31;
	add.s32 	%r17, %r12, %r16;
	shr.s32 	%r3, %r17, 1;
	mov.u32 	%r18, %nctaid.x;
	mul.lo.s32 	%r4, %r18, %r25;

$L__BB2_2:
	setp.ge.s32 	%p2, %r2, %r3;
	@%p2 bra 	$L__BB2_9;

	shl.b32 	%r6, %r25, 1;
	mov.u32 	%r26, %r2;
	bra.uni 	$L__BB2_4;

$L__BB2_6:
	setp.le.s32 	%p5, %r8, %r9;
	@%p5 bra 	$L__BB2_8;
	bra.uni 	$L__BB2_7;

$L__BB2_4:
	div.s32 	%r19, %r26, %r25;
	mul.lo.s32 	%r20, %r19, %r25;
	sub.s32 	%r21, %r26, %r20;
	mad.lo.s32 	%r22, %r6, %r19, %r21;
	xor.b32  	%r23, %r22, %r25;
	and.b32  	%r24, %r22, %r13;
	setp.eq.s32 	%p3, %r24, 0;
	mul.wide.s32 	%rd5, %r22, 4;
	add.s64 	%rd2, %rd1, %rd5;
	ld.global.u32 	%r8, [%rd2];
	mul.wide.s32 	%rd6, %r23, 4;
	add.s64 	%rd3, %rd1, %rd6;
	ld.global.u32 	%r9, [%rd3];
	@%p3 bra 	$L__BB2_6;

	setp.lt.s32 	%p4, %r8, %r9;
	@%p4 bra 	$L__BB2_7;
	bra.uni 	$L__BB2_8;

$L__BB2_7:
	st.global.u32 	[%rd2], %r9;
	st.global.u32 	[%rd3], %r8;

$L__BB2_8:
	add.s32 	%r26, %r26, %r4;
	setp.lt.s32 	%p6, %r26, %r3;
	@%p6 bra 	$L__BB2_4;

$L__BB2_9:
	bar.sync 	0;
	shr.s32 	%r11, %r25, 1;
	setp.gt.s32 	%p7, %r25, 1;
	mov.u32 	%r25, %r11;
	@%p7 bra 	$L__BB2_2;

$L__BB2_10:
	ret;

}

`