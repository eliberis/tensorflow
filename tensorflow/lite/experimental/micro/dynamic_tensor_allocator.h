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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_DYNAMIC_TENSOR_ALLOCATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_DYNAMIC_TENSOR_ALLOCATOR_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

class DynamicTensorAllocator {
 public:
  DynamicTensorAllocator(TfLiteContext *context, uint8_t* buffer, size_t buffer_size)
      : context_(context), front_offset_(0), back_offset_(buffer_size),
        data_size_max_(buffer_size), data_(buffer), num_allocated_tensors_(0), needs_compacting_(false) {}

  // Allocates memory for a tensor.
  TfLiteStatus AllocateBuffer(TfLiteTensor *tensor, ErrorReporter* error_reporter);

  // Deallocates memory used by a tensor.
  TfLiteStatus DeallocateBuffer(TfLiteTensor *tensor, ErrorReporter* error_reporter);

  // Permanently allocates a chunk of memory (cannot be deallocated). This is
  // intended for data that is expected to live as long as the interpreter
  // (e.g. tensor metadata buffers). These buffers will not participate in memory
  // management and reclamation logic and hence won't slow it down.
  // In this implementation, static data is allocated from the end (back) of the buffer.
  TfLiteStatus AllocateStaticMemory(size_t size, size_t alignment,
          ErrorReporter* error_reporter, uint8_t **output);

  int GetDataSize() const { return data_size_max_ - (back_offset_ - front_offset_); }

 private:
  TfLiteStatus Compact(ErrorReporter *error_reporter);

//  struct Chunk {
//    TfLiteTensor *tensor;
//    size_t offset;
//  };

  static const int kMaxAllocatedTensors = 32;
  TfLiteTensor* allocated_tensors_[kMaxAllocatedTensors];
  int num_allocated_tensors_;

  TfLiteContext *context_;
  size_t front_offset_, back_offset_;
  size_t data_size_max_;
  uint8_t* data_;
  bool needs_compacting_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_DYNAMIC_TENSOR_ALLOCATOR_H_
