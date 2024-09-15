package cuda

//#include <cuda.h>
import "C"
import (
	"math"
	"unsafe"
)

type Device struct {
	dev C.CUdevice
}

type FlushGPUDirectRDMAWritesScope int
type FlushGPUDirectRDMAWritesTarget int
type AffinityType uint32
type DeviceAttribute uint32

func DeviceCount() (int, Result) {
	var count C.int
	err := C.cuDeviceGetCount(&count)
	if err != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(err))
	}
	return int(count), nil
}

func DeviceGet(device int) (*Device, Result) {
	var dev C.CUdevice
	err := C.cuDeviceGet(&dev, C.int(device))
	if err != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(err))
	}
	return &Device{dev}, nil
}

func (dev *Device) Name() (string, Result) {
	defaultSize := 256
	name := (*C.char)(C.malloc(C.ulong(defaultSize)))
	defer C.free(unsafe.Pointer(name))
	err := C.cuDeviceGetName(name, C.int(defaultSize), dev.dev)
	if err != C.CUDA_SUCCESS {
		return "", NewCudaError(uint32(err))
	}
	return C.GoString(name), nil
}

func (dev *Device) TotalMem() (uint64, Result) {
	var mem C.size_t
	err := C.cuDeviceTotalMem(&mem, dev.dev)
	if err != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(err))
	}
	return uint64(mem), nil
}

func (dev *Device) GetDefaultMemPool() {
	//TODO: implement
}

func (dev *Device) GetMemPool() {
	//TODO: implement
}

func (dev *Device) SetMemPool() Result {
	//TODO: implement
	return ErrUnsupported
}

func (dev *Device) UUID() (string, Result) {
	uuid := make([]byte, 16)
	err := C.cuDeviceGetUuid((*C.CUuuid)(unsafe.Pointer(&uuid[0])), dev.dev)
	if err != C.CUDA_SUCCESS {
		return "", NewCudaError(uint32(err))
	}
	return string(uuid), nil
}

func (dev *Device) UUIDv2() (string, Result) {
	uuid := make([]byte, 16)
	err := C.cuDeviceGetUuid_v2((*C.CUuuid)(unsafe.Pointer(&uuid[0])), dev.dev)
	if err != C.CUDA_SUCCESS {
		return "", NewCudaError(uint32(err))
	}
	return string(uuid), nil
}

func (dev *Device) GetAttribute(attr DeviceAttribute) (int, Result) {
	var attri C.int
	err := C.cuDeviceGetAttribute(&attri, C.CUdevice_attribute(attr), dev.dev)
	if err != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(err))
	}
	return int(attr), nil
}

// ignores error and returns int.Min if error
func (dev *Device) fastGetAttr(attr DeviceAttribute) int {
	var attri C.int = math.MinInt32
	C.cuDeviceGetAttribute(&attri, C.CUdevice_attribute(attr), dev.dev)
	return int(attr)
}

func (dev *Device) GetAllAttributes() map[DeviceAttribute]int {
	attrs := make(map[DeviceAttribute]int)
	for attrNum := range deviceAttributeStrings {
		attrs[attrNum] = dev.fastGetAttr(attrNum)
	}
	return attrs
}

func (dev *Device) GetExecAffinitySupport(afType AffinityType) (int, Result) {
	var supported C.int
	err := C.cuDeviceGetAttribute(&supported, C.CUdevice_attribute(afType), dev.dev)
	if err != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(err))
	}
	return int(supported), nil
}

// GetLuid returns (LUID, deviceNodeMask) for the device
func (dev *Device) GetLuid() (byte, uint32, Result) {
	//TODO: implement
	/*var luid C.CUuuid
	var nodeMask C.uint
	err := C.cuDeviceGetLuid(&luid, &nodeMask, dev.dev)
	if err != C.CUDA_SUCCESS {
		return 0, 0, NewCudaError(uint32(err))
	}
	return byte(luid), uint32(nodeMask), nil*/
	return 0, 0, ErrUnsupported
}

func (dev *Device) GetNvSciSyncAttributes() {
	//TODO: implement
}

func (dev *Device) GetTexture1SLinearMaxWidth(format int, numChannels uint32) (uint64, Result) {
	//TODO: implement till end
	return 0, ErrUnsupported
}

func (dev *Device) NativeHandle() uintptr {
	return uintptr(dev.dev)
}

func FlushGPUDirectRDMAWrites(target FlushGPUDirectRDMAWritesTarget,
	scope FlushGPUDirectRDMAWritesScope) Result {
	err := C.cuFlushGPUDirectRDMAWrites(C.CUflushGPUDirectRDMAWritesTarget(target),
		C.CUflushGPUDirectRDMAWritesScope(scope))
	if err != C.CUDA_SUCCESS {
		return NewCudaError(uint32(err))
	}
	return nil
}

const (
	CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX FlushGPUDirectRDMAWritesTarget = 0 // Sets the target for FlushGPUDirectRDMAWrites() to the currently active CUDA device context.

	CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER       FlushGPUDirectRDMAWritesScope = 100 // Blocks until remote writes are visible to the CUDA device context owning the data.
	CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES FlushGPUDirectRDMAWritesScope = 200 // Blocks until remote writes are visible to all CUDA device contexts.
)

const (
	CU_EXEC_AFFINITY_TYPE_SM_COUNT AffinityType = 0 //1 if context with limited SMs is supported by the device, or 0 if not;
)

const (
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                        DeviceAttribute = 1   // Maximum number of threads per block
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                              DeviceAttribute = 2   // Maximum block dimension X
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                              DeviceAttribute = 3   // Maximum block dimension Y
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                              DeviceAttribute = 4   // Maximum block dimension Z
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                               DeviceAttribute = 5   // Maximum grid dimension X
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                               DeviceAttribute = 6   // Maximum grid dimension Y
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                               DeviceAttribute = 7   // Maximum grid dimension Z
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK                  DeviceAttribute = 8   // Maximum shared memory available per block in bytes
	CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK                      DeviceAttribute = 8   // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
	CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY                        DeviceAttribute = 9   // Memory available on device for __constant__ variables in a CUDA C kernel in bytes
	CU_DEVICE_ATTRIBUTE_WARP_SIZE                                    DeviceAttribute = 10  // Warp size in threads
	CU_DEVICE_ATTRIBUTE_MAX_PITCH                                    DeviceAttribute = 11  // Maximum pitch in bytes allowed by memory copies
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK                      DeviceAttribute = 12  // Maximum number of 32-bit registers available per block
	CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK                          DeviceAttribute = 12  // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
	CU_DEVICE_ATTRIBUTE_CLOCK_RATE                                   DeviceAttribute = 13  // Typical clock frequency in kilohertz
	CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                            DeviceAttribute = 14  // Alignment requirement for textures
	CU_DEVICE_ATTRIBUTE_GPU_OVERLAP                                  DeviceAttribute = 15  // Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.
	CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                         DeviceAttribute = 16  // Number of multiprocessors on device
	CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT                          DeviceAttribute = 17  // Specifies whether there is a run time limit on kernels
	CU_DEVICE_ATTRIBUTE_INTEGRATED                                   DeviceAttribute = 18  // Device is integrated with host memory
	CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                          DeviceAttribute = 19  // Device can map host memory into CUDA address space
	CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                                 DeviceAttribute = 20  // Compute mode (See CUcomputemode for details)
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH                      DeviceAttribute = 21  // Maximum 1D texture width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH                      DeviceAttribute = 22  // Maximum 2D texture width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT                     DeviceAttribute = 23  // Maximum 2D texture height
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH                      DeviceAttribute = 24  // Maximum 3D texture width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT                     DeviceAttribute = 25  // Maximum 3D texture height
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH                      DeviceAttribute = 26  // Maximum 3D texture depth
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH              DeviceAttribute = 27  // Maximum 2D layered texture width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT             DeviceAttribute = 28  // Maximum 2D layered texture height
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS             DeviceAttribute = 29  // Maximum layers in a 2D layered texture
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH                DeviceAttribute = 27  // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT               DeviceAttribute = 28  // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES            DeviceAttribute = 29  // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
	CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT                            DeviceAttribute = 30  // Alignment requirement for surfaces
	CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS                           DeviceAttribute = 31  // Device can possibly execute multiple kernels concurrently
	CU_DEVICE_ATTRIBUTE_ECC_ENABLED                                  DeviceAttribute = 32  // Device has ECC support enabled
	CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                                   DeviceAttribute = 33  // PCI bus ID of the device
	CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                                DeviceAttribute = 34  // PCI device ID of the device
	CU_DEVICE_ATTRIBUTE_TCC_DRIVER                                   DeviceAttribute = 35  // Device is using TCC driver model
	CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                            DeviceAttribute = 36  // Peak memory clock frequency in kilohertz
	CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH                      DeviceAttribute = 37  // Global memory bus width in bits
	CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                                DeviceAttribute = 38  // Size of L2 cache in bytes
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR               DeviceAttribute = 39  // Maximum resident threads per multiprocessor
	CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT                           DeviceAttribute = 40  // Number of asynchronous engines
	CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING                           DeviceAttribute = 41  // Device shares a unified address space with the host
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH              DeviceAttribute = 42  // Maximum 1D layered texture width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS             DeviceAttribute = 43  // Maximum layers in a 1D layered texture
	CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER                             DeviceAttribute = 44  // Deprecated, do not use.
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH               DeviceAttribute = 45  // Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT              DeviceAttribute = 46  // Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE            DeviceAttribute = 47  // Alternate maximum 3D texture width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE           DeviceAttribute = 48  // Alternate maximum 3D texture height
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE            DeviceAttribute = 49  // Alternate maximum 3D texture depth
	CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                                DeviceAttribute = 50  // PCI domain ID of the device
	CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT                      DeviceAttribute = 51  // Pitch alignment requirement for textures
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH                 DeviceAttribute = 52  // Maximum cubemap texture width/height
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH         DeviceAttribute = 53  // Maximum cubemap layered texture width/height
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS        DeviceAttribute = 54  // Maximum layers in a cubemap layered texture
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH                      DeviceAttribute = 55  // Maximum 1D surface width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH                      DeviceAttribute = 56  // Maximum 2D surface width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT                     DeviceAttribute = 57  // Maximum 2D surface height
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH                      DeviceAttribute = 58  // Maximum 3D surface width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT                     DeviceAttribute = 59  // Maximum 3D surface height
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH                      DeviceAttribute = 60  // Maximum 3D surface depth
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH              DeviceAttribute = 61  // Maximum 1D layered surface width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS             DeviceAttribute = 62  // Maximum layers in a 1D layered surface
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH              DeviceAttribute = 63  // Maximum 2D layered surface width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT             DeviceAttribute = 64  // Maximum 2D layered surface height
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS             DeviceAttribute = 65  // Maximum layers in a 2D layered surface
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH                 DeviceAttribute = 66  // Maximum cubemap surface width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH         DeviceAttribute = 67  // Maximum cubemap layered surface width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS        DeviceAttribute = 68  // Maximum layers in a cubemap layered surface
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH               DeviceAttribute = 69  // Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH               DeviceAttribute = 70  // Maximum 2D linear texture width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT              DeviceAttribute = 71  // Maximum 2D linear texture height
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH               DeviceAttribute = 72  // Maximum 2D linear texture pitch in bytes
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH            DeviceAttribute = 73  // Maximum mipmapped 2D texture width
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT           DeviceAttribute = 74  // Maximum mipmapped 2D texture height
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR                     DeviceAttribute = 75  // Major compute capability version number
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR                     DeviceAttribute = 76  // Minor compute capability version number
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH            DeviceAttribute = 77  // Maximum mipmapped 1D texture width
	CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED                  DeviceAttribute = 78  // Device supports stream priorities
	CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED                    DeviceAttribute = 79  // Device supports caching globals in L1
	CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED                     DeviceAttribute = 80  // Device supports caching locals in L1
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR         DeviceAttribute = 81  // Maximum shared memory available per multiprocessor in bytes
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR             DeviceAttribute = 82  // Maximum number of 32-bit registers available per multiprocessor
	CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY                               DeviceAttribute = 83  // Device can allocate managed memory on this system
	CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD                              DeviceAttribute = 84  // Device is on a multi-GPU board
	CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID                     DeviceAttribute = 85  // Unique id for a group of devices on the same multi-GPU board
	CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED                 DeviceAttribute = 86  // Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)
	CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO        DeviceAttribute = 87  // Ratio of single precision performance (in floating-point operations per second) to double precision performance
	CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS                       DeviceAttribute = 88  // Device supports coherently accessing pageable memory without calling cudaHostRegister on it
	CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS                    DeviceAttribute = 89  // Device can coherently access managed memory concurrently with the CPU
	CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED                 DeviceAttribute = 90  // Device supports compute preemption.
	CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM      DeviceAttribute = 91  // Device can access host registered memory at the same virtual address as the CPU
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1                    DeviceAttribute = 92  // Deprecated, along with v1 MemOps API, cuStreamBatchMemOp and related APIs are supported.
	CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1             DeviceAttribute = 93  // Deprecated, along with v1 MemOps API, 64-bit operations are supported in cuStreamBatchMemOp and related APIs.
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1             DeviceAttribute = 94  // Deprecated, along with v1 MemOps API, CU_STREAM_WAIT_VALUE_NOR is supported.
	CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH                           DeviceAttribute = 95  // Device supports launching cooperative kernels via cuLaunchCooperativeKernel
	CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH              DeviceAttribute = 96  // Deprecated, cuLaunchCooperativeKernelMultiDevice is deprecated.
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN            DeviceAttribute = 97  // Maximum optin shared memory per block
	CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES                      DeviceAttribute = 98  // The CU_STREAM_WAIT_VALUE_FLUSH flag and the CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See Stream Memory Operations for additional details.
	CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED                      DeviceAttribute = 99  // Device supports host memory registration via cudaHostRegister.
	CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES DeviceAttribute = 100 // Device accesses pageable memory via the host's page tables.
	CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST          DeviceAttribute = 101 // The host can directly access managed memory on the device without migration.
	CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED         DeviceAttribute = 102 // Deprecated, Use CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
	CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED          DeviceAttribute = 102 // Device supports virtual memory management APIs like cuMemAddressReserve, cuMemCreate, cuMemMap and related APIs
	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED  DeviceAttribute = 103 // Device supports exporting memory to a posix file descriptor with cuMemExportToShareableHandle, if requested via cuMemCreate
	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED           DeviceAttribute = 104 // Device supports exporting memory to a Win32 NT handle with cuMemExportToShareableHandle, if requested via cuMemCreate
	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED       DeviceAttribute = 105 // Device supports exporting memory to a Win32 KMT handle with cuMemExportToShareableHandle, if requested via cuMemCreate
	CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR                DeviceAttribute = 106 // Maximum number of blocks per multiprocessor
	CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED                DeviceAttribute = 107 // Device supports compression of memory
	CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE                 DeviceAttribute = 108 // Maximum L2 persisting lines capacity setting in bytes.
	CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE                DeviceAttribute = 109 // Maximum value of CUaccessPolicyWindow::num_bytes.
	CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED      DeviceAttribute = 110 // Device supports specifying the GPUDirect RDMA flag with cuMemCreate
	CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK             DeviceAttribute = 111 // Shared memory reserved by CUDA driver per block in bytes
	CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED                  DeviceAttribute = 112 // Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
	CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED            DeviceAttribute = 113 // Device supports using the cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU
	CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED         DeviceAttribute = 114 // External timeline semaphore interop is supported on the device
	CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED                       DeviceAttribute = 115 // Device supports using the cuMemAllocAsync and cuMemPool family of APIs
	CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED                    DeviceAttribute = 116 // Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)
	CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS         DeviceAttribute = 117 // The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the CUflushGPUDirectRDMAWritesOptions enum
	CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING              DeviceAttribute = 118 // GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
	CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES               DeviceAttribute = 119 // Handle types supported with mempool based IPC
	CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH                               DeviceAttribute = 120 // Indicates device supports cluster launch
	CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED        DeviceAttribute = 121 // Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
	CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS                DeviceAttribute = 122 // 64-bit operations are supported in cuStreamBatchMemOp and related MemOp APIs.
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR                DeviceAttribute = 123 //  CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs.
	CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED                            DeviceAttribute = 124 // Device supports buffer sharing with dma_buf mechanism.
	CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED                          DeviceAttribute = 125 // Device supports IPC Events.
	CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT                        DeviceAttribute = 126 // Number of memory domains the device supports.
	CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED                  DeviceAttribute = 127 // Device supports accessing memory using Tensor Map.
	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED                 DeviceAttribute = 128 // Device supports exporting memory to a fabric handle with cuMemExportToShareableHandle() or requested with cuMemCreate()
	CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS                    DeviceAttribute = 129 // Device supports unified function pointers.
	CU_DEVICE_ATTRIBUTE_NUMA_CONFIG                                  DeviceAttribute = 130 // NUMA configuration of a device: value is of type CUdeviceNumaConfig enum
	CU_DEVICE_ATTRIBUTE_NUMA_ID                                      DeviceAttribute = 131 // NUMA node ID of the GPU memory
	CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED                          DeviceAttribute = 132 // Device supports switch multicast and reduction operations.
	CU_DEVICE_ATTRIBUTE_MPS_ENABLED                                  DeviceAttribute = 133 // Indicates if contexts created on this device will be shared via MPS
	CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID                                 DeviceAttribute = 134 // NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA.
	CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED                          DeviceAttribute = 135 // Device supports CIG with D3D12.
	CU_DEVICE_ATTRIBUTE_MAX                                          DeviceAttribute = 136 // Sentinel value to indicate the last value when querying for device attributes.
)

var deviceAttributeStrings = map[DeviceAttribute]string{
	1:   "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
	2:   "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X",
	3:   "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y",
	4:   "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z",
	5:   "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X",
	6:   "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y",
	7:   "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z",
	8:   "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK",
	9:   "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY",
	10:  "CU_DEVICE_ATTRIBUTE_WARP_SIZE",
	11:  "CU_DEVICE_ATTRIBUTE_MAX_PITCH",
	12:  "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK",
	13:  "CU_DEVICE_ATTRIBUTE_CLOCK_RATE",
	14:  "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT",
	15:  "CU_DEVICE_ATTRIBUTE_GPU_OVERLAP",
	16:  "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",
	17:  "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT",
	18:  "CU_DEVICE_ATTRIBUTE_INTEGRATED",
	19:  "CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY",
	20:  "CU_DEVICE_ATTRIBUTE_COMPUTE_MODE",
	21:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH",
	22:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH",
	23:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT",
	24:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH",
	25:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT",
	26:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH",
	27:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH",
	28:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT",
	29:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS",
	30:  "CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT",
	31:  "CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS",
	32:  "CU_DEVICE_ATTRIBUTE_ECC_ENABLED",
	33:  "CU_DEVICE_ATTRIBUTE_PCI_BUS_ID",
	34:  "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID",
	35:  "CU_DEVICE_ATTRIBUTE_TCC_DRIVER",
	36:  "CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE",
	37:  "CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH",
	38:  "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE",
	39:  "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR",
	40:  "CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT",
	41:  "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING",
	42:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH",
	43:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS",
	45:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH",
	46:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT",
	47:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE",
	48:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE",
	49:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE",
	50:  "CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID",
	51:  "CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT",
	52:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH",
	53:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH",
	54:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS",
	55:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH",
	56:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH",
	57:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT",
	58:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH",
	59:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT",
	60:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH",
	61:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH",
	62:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS",
	63:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH",
	64:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT",
	65:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS",
	66:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH",
	67:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH",
	68:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS",
	70:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH",
	71:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT",
	72:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH",
	73:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH",
	74:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT",
	75:  "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
	76:  "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
	77:  "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH",
	78:  "CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED",
	79:  "CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED",
	80:  "CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED",
	81:  "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
	82:  "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR",
	83:  "CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY",
	84:  "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD",
	85:  "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID",
	86:  "CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED",
	87:  "CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO",
	88:  "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS",
	89:  "CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS",
	90:  "CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED",
	91:  "CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM",
	95:  "CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH",
	97:  "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
	98:  "CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES",
	99:  "CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED",
	100: "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES",
	101: "CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST",
	102: "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
	103: "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED",
	104: "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED",
	105: "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED",
	106: "CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR",
	107: "CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED",
	108: "CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE",
	109: "CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE",
	110: "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED",
	111: "CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK",
	112: "CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED",
	113: "CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED",
	114: "CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED",
	115: "CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED",
	116: "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED",
	117: "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS",
	118: "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING",
	119: "CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES",
	120: "CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH",
	121: "CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED",
	122: "CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS",
	123: "CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR",
	124: "CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED",
	125: "CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED",
	126: "CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT",
	127: "CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED",
	128: "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED",
	129: "CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS",
	130: "CU_DEVICE_ATTRIBUTE_NUMA_CONFIG",
	131: "CU_DEVICE_ATTRIBUTE_NUMA_ID",
	132: "CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED",
	133: "CU_DEVICE_ATTRIBUTE_MPS_ENABLED",
	134: "CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID",
	135: "CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED",
}
