/** Prints CUDA GPU information in a machine-readable user-friendly format.
 *
 * The output can be read with a YAML parser, and is an array with one element
 * per CUDA GPU.
 *
 * Build with:
 *     nvcc -o cudainfo cudainfo.cu
 */
#include <stdio.h>

int main() {
  cudaDeviceProp deviceProperties;
  cudaError_t status;
  int deviceCount;

  if((status = cudaGetDeviceCount(&deviceCount)) != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    return 1;
  }

  printf("---\n");
  for (int i = 0; i < deviceCount; ++i) {
    status = cudaGetDeviceProperties(&deviceProperties, i);
    if(status != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
      return 1;
    }

    printf("- name: \"%s\"\n", deviceProperties.name);
    printf("  compute_version: \"%d.%d\"\n",
           deviceProperties.major, deviceProperties.minor);
    printf("  pci_address: \"%02d:%02d\"\n",
           deviceProperties.pciBusID, deviceProperties.pciDeviceID);

    printf("  total_global_memory: %zu\n", deviceProperties.totalGlobalMem);
    printf("  total_constant_memory: %zu\n", deviceProperties.totalConstMem);
    printf("  shared_memory_per_block: %zu\n",
           deviceProperties.sharedMemPerBlock);
    printf("  max_malloc_pitch: %zu\n", deviceProperties.memPitch);
    printf("  texture_alignment: %zu\n", deviceProperties.textureAlignment);

    printf("  registers_per_block: %d\n", deviceProperties.regsPerBlock);
    printf("  max_threads_per_block: %d\n",
           deviceProperties.maxThreadsPerBlock);
    printf("  max_thread_block_dimension: [%d, %d, %d]\n",
           deviceProperties.maxThreadsDim[0], deviceProperties.maxThreadsDim[1],
           deviceProperties.maxThreadsDim[2]);
    printf("  max_grid_size: [%d, %d, %d]\n",
           deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1],
           deviceProperties.maxGridSize[2]);
    printf("  warp_size_threads: %d\n", deviceProperties.warpSize);

    printf("  multi_processor_count: %d\n",
           deviceProperties.multiProcessorCount);
    printf("  clock_rate_khz: %d\n", deviceProperties.clockRate);
    printf("  pci_bus_id: %d\n", deviceProperties.pciBusID);
    printf("  pci_device_id: %d\n", deviceProperties.pciDeviceID);
    printf("  compute_major: %d\n", deviceProperties.major);
    printf("  compute_minor: %d\n", deviceProperties.minor);

    printf("  integrated: %s\n",
           deviceProperties.integrated ? "true" : "false");
    printf("  supports_device_overlap: %s\n",
           deviceProperties.deviceOverlap ? "true" : "false");
    printf("  kernel_execution_timeout_enabled: %s\n",
           deviceProperties.kernelExecTimeoutEnabled ? "true" : "false");
    printf("  can_map_host_memory: %s\n",
           deviceProperties.canMapHostMemory ? "true" : "false");
    printf("  supports_concurrent_kernels: %s\n",
           deviceProperties.concurrentKernels ? "true" : "false");
    printf("  ecc_enabled: %s\n",
           deviceProperties.ECCEnabled ? "true" : "false");
    printf("  using_tcc_driver: %s\n",
           deviceProperties.tccDriver ? "true" : "false");

    const char* computeMode;
    switch (deviceProperties.computeMode) {
    case cudaComputeModeDefault:
      computeMode = "default";
      break;
    case cudaComputeModeExclusive:
      computeMode = "exclusive";
      break;
    case cudaComputeModeProhibited:
      computeMode = "prohibited";
      break;
    default:
      computeMode = "unknown";
    }
    printf("  compute_mode: %s\n", computeMode);
  }

  return 0;
}