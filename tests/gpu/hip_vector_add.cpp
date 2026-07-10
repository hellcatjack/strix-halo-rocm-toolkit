#include <hip/hip_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t error = call;                                                  \
    if (error != hipSuccess) {                                                \
      std::cerr << hipGetErrorString(error) << std::endl;                     \
      return 2;                                                               \
    }                                                                         \
  } while (0)

__global__ void add(const float* left, const float* right, float* output,
                    int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = left[index] + right[index];
  }
}

int main() {
  constexpr int count = 1 << 20;
  const size_t bytes = count * sizeof(float);
  std::vector<float> left(count, 1.25f);
  std::vector<float> right(count, 2.5f);
  std::vector<float> output(count);
  float* device_left = nullptr;
  float* device_right = nullptr;
  float* device_output = nullptr;
  HIP_CHECK(hipMalloc(&device_left, bytes));
  HIP_CHECK(hipMalloc(&device_right, bytes));
  HIP_CHECK(hipMalloc(&device_output, bytes));
  HIP_CHECK(
      hipMemcpy(device_left, left.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(device_right, right.data(), bytes, hipMemcpyHostToDevice));
  add<<<(count + 255) / 256, 256>>>(device_left, device_right, device_output,
                                    count);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(
      hipMemcpy(output.data(), device_output, bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(device_left));
  HIP_CHECK(hipFree(device_right));
  HIP_CHECK(hipFree(device_output));
  for (float value : output) {
    if (std::fabs(value - 3.75f) > 1e-6f) {
      return 3;
    }
  }
  std::cout << "HIP vector add PASS" << std::endl;
  return 0;
}
