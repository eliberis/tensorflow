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
#include "tensorflow/lite/experimental/micro/dynamic_tensor_allocator.h"

namespace tflite {

TfLiteStatus DynamicTensorAllocator::AllocateBuffer(TfLiteTensor *tensor,
                                                    ErrorReporter *error_reporter) {
  if (needs_compacting_) {
    TF_LITE_ENSURE_STATUS(Compact(error_reporter));
  }

  if (num_allocated_tensors_ == kMaxAllocatedTensors) {
    error_reporter->Report("Exceeded the maximum number of tensors "
                           "that can be allocated (%d).", kMaxAllocatedTensors);
    return kTfLiteError;
  }

  size_t type_size;
  TF_LITE_ENSURE_STATUS(TfLiteTypeSizeOf(tensor->type, &type_size, error_reporter));

  uint8_t *tensor_data = AlignPointerRoundUp(data_ + front_offset_, type_size);

  if (tensor_data + tensor->bytes > data_ + back_offset_) {
    error_reporter->Report("Failed to allocate memory for tensor '%s'. Out of memory.", tensor->name);
  }

  allocated_tensors_[num_allocated_tensors_] = tensor;
  num_allocated_tensors_++;
  tensor->data.uint8 = tensor_data;
  front_offset_ = (tensor_data - data_) + tensor->bytes;

  return kTfLiteOk;
}

TfLiteStatus DynamicTensorAllocator::DeallocateBuffer(TfLiteTensor *tensor,
                                                      ErrorReporter *error_reporter) {
  // Find the tensor in the allocation table
  int idx = 0;
  while (idx < num_allocated_tensors_ && allocated_tensors_[idx] != tensor) {
    idx++;
  }

  if (idx == num_allocated_tensors_) {
    error_reporter->Report("Failed to deallocate tensor '%s'; the tensor "
                           "hasn't been allocated in the first place.", tensor->name);
    return kTfLiteError;
  }

  // Skip a trivial case where the tensor being deallocated is the last one
  if (idx - 1 != num_allocated_tensors_) {
    // Delete the entry corresponding to the tensor
    for (int i = idx; i < num_allocated_tensors_ - 1; i++) {
      allocated_tensors_[i] = allocated_tensors_[i + 1];
    }
    needs_compacting_ = true;
  }
  num_allocated_tensors_--;

  tensor->data.uint8 = nullptr;
  return kTfLiteOk;
}

TfLiteStatus DynamicTensorAllocator::AllocateStaticMemory(size_t size, size_t alignment,
                                                          ErrorReporter* error_reporter, uint8_t **output) {
  uint8_t* current_data = data_ + back_offset_;
  uint8_t* aligned_result = AlignPointerRoundDown(current_data - size, alignment);

  if (aligned_result < data_ + front_offset_) {
    error_reporter->Report("Failed to allocate memory: wanted %d bytes, "
                           "but only %d were available.", size, back_offset_ - front_offset_);
    *output = nullptr;
    return kTfLiteError;
  }

  *output = aligned_result;
  back_offset_ = aligned_result - data_;
  return kTfLiteOk;
}

TfLiteStatus DynamicTensorAllocator::Compact(ErrorReporter *error_reporter) {
  needs_compacting_ = false;

  front_offset_ = 0;
  for (int i = 0; i < num_allocated_tensors_; i++) {
    TfLiteTensor *tensor = allocated_tensors_[i];
    size_t type_size;
    TF_LITE_ENSURE_STATUS(TfLiteTypeSizeOf(tensor->type, &type_size, error_reporter));

    uint8_t *tensor_ptr = AlignPointerRoundUp(data_ + front_offset_, type_size);
    if (tensor_ptr != tensor->data.uint8) {
      MoveBuffers(tensor_ptr, tensor->data.uint8, tensor->bytes);
      tensor->data.uint8 = tensor_ptr;
    }
    front_offset_ = (tensor_ptr - data_) + tensor->bytes;
  }

  return kTfLiteOk;
}

}  // namespace tflite
