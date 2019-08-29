/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/experimental/micro/allocator_utils.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"

#if __ANDROID__ || defined(__x86_64__) || defined(__i386__) || \
    defined(__i386) || defined(__x86__) || defined(__X86__) || \
    defined(_X86_) || defined(_M_IX86) || defined(_M_X64)
#include <cstring>
#define HAS_STDLIB_MEMMOVE 1
#else
#define HAS_STDLIB_MEMMOVE 0
#endif

namespace tflite {

uint8_t* AlignPointerRoundUp(uint8_t* data, size_t alignment) {
  size_t data_as_size_t = reinterpret_cast<size_t>(data);
  uint8_t* aligned_result = reinterpret_cast<uint8_t*>(
          ((data_as_size_t + (alignment - 1)) / alignment) * alignment);
  return aligned_result;
}

uint8_t* AlignPointerRoundDown(uint8_t* data, size_t alignment) {
  size_t data_as_size_t = reinterpret_cast<size_t>(data);
  uint8_t* aligned_result = reinterpret_cast<uint8_t*>(
          (data_as_size_t / alignment) * alignment);
  return aligned_result;
}

TfLiteStatus TfLiteTypeSizeOf(TfLiteType type, size_t* size,
                              ErrorReporter* reporter) {
  switch (type) {
    case kTfLiteFloat32:
      *size = sizeof(float);
      break;
    case kTfLiteInt16:
      *size = sizeof(int16_t);
      break;
    case kTfLiteInt32:
      *size = sizeof(int32_t);
      break;
    case kTfLiteInt8:
      *size = sizeof(int8_t);
      break;
    case kTfLiteUInt8:
      *size = sizeof(uint8_t);
      break;
    case kTfLiteInt64:
      *size = sizeof(int64_t);
      break;
    case kTfLiteBool:
      *size = sizeof(bool);
      break;
    case kTfLiteComplex64:
      *size = sizeof(float) * 2;
      break;
    default:
      reporter->Report("Type %s (%d) not is not supported",
                       TfLiteTypeGetName(type), type);
      return kTfLiteError;
    }
  return kTfLiteOk;
}

TfLiteStatus BytesRequired(const tflite::Tensor& flatbuffer_tensor,
                           size_t dims_size, size_t* bytes, size_t* type_size,
                           ErrorReporter* error_reporter) {
  TfLiteType tf_lite_type;
  TF_LITE_ENSURE_STATUS(ConvertTensorType(flatbuffer_tensor.type(),
                                          &tf_lite_type, error_reporter));
  TF_LITE_ENSURE_STATUS(
          TfLiteTypeSizeOf(tf_lite_type, type_size, error_reporter));
  *bytes = dims_size * (*type_size);
  return kTfLiteOk;
}

void MoveBuffers(void *dest, const void *src, size_t n) {
#if HAS_STDLIB_MEMMOVE
  memmove(dest, src, n);
#else
  char* destination = reinterpret_cast<char*>(dest);
  const char* source = reinterpret_cast<const char*>(src);
  if (destination < source) {
    // Copy forward
    for (size_t i = 0; i < n; i++) {
      destination[i] = source[i];
    }
  } else {
    // Copy backwards
    for (size_t i = n; i > 0; i--) {
      destination[i - 1] = source[i - 1];
    }
  }
#endif
}

} // namespace tflite

