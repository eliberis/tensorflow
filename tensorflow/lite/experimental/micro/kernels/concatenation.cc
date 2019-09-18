/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace concatenation {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
//  auto* params =
//      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);
//  int axis = params->axis;
//  int num_inputs = node->inputs->size;
//
//  // The number of dimensions of the input tensors must match, and all
//  // dimensions except 'axis' must be equal.
//  TfLiteTensor* t0 = &context->tensors[node->inputs->data[0]];
//  TfLiteType input_type = t0->type;
//  if (axis < 0) axis += t0->dims->size;
//  TF_LITE_ENSURE(context, axis >= 0);
//  TF_LITE_ENSURE(context, axis < t0->dims->size);
//
//  // TODO(ahentz): These are limitations of our implementation that could be
//  // removed with a bit of effort.
//  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);
//  TF_LITE_ENSURE(context,
//                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
//                     input_type == kTfLiteInt8 || input_type == kTfLiteInt16 ||
//                     input_type == kTfLiteInt32 || input_type == kTfLiteInt64);
//
//  // Output dimensions will match input dimensions, except 'axis', which
//  // will be the sum of inputs
//  int sum_axis = t0->dims->data[axis];
//  for (int i = 1; i < num_inputs; ++i) {
//    TfLiteTensor* t = &context->tensors[node->inputs->data[i]];
//    TF_LITE_ENSURE_EQ(context, t->dims->size, t0->dims->size);
//    TF_LITE_ENSURE_EQ(context, t->type, input_type);
//    for (int d = 0; d < t0->dims->size; ++d) {
//      if (d == axis) {
//        sum_axis += t->dims->data[axis];
//      } else {
//        TF_LITE_ENSURE_EQ(context, t->dims->data[d], t0->dims->data[d]);
//      }
//    }
//  }
//
//  TfLiteIntArray* output_size = t0->dims;
//  for (int d = 0; d < t0->dims->size; ++d) {
//    output_size->data[d] = (d == axis) ? sum_axis : t0->dims->data[d];
//  }
//
//  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
//  TF_LITE_ENSURE_EQ(context, output->type, input_type);
//
//  if (input_type == kTfLiteInt8) {
//    // Make sure there is no re-scaling needed for Int8 quantized kernel. This
//    // is a restriction we introduced to Int8 kernels.
//    VectorOfTensors<int8_t> all_inputs(*context, *node->inputs);
//    for (int i = 0; i < node->inputs->size; ++i) {
//      TfLiteTensor* t = &context->tensors[node->inputs->data[i]];
//      TF_LITE_ENSURE_EQ(context, t->params.scale, output->params.scale);
//      TF_LITE_ENSURE_EQ(context, t->params.zero_point,
//                        output->params.zero_point);
//    }
//  }
//
//  return context->ResizeTensor(context, output, output_size);
}

template <typename Scalar>
inline void Concatenation(const ConcatenationParams& params,
                          const RuntimeShape* const* input_shapes,
                          const Scalar* const* input_data,
                          const RuntimeShape& output_shape,
                          Scalar* output_data) {
  int axis = params.axis;
  int inputs_count = params.inputs_count;
  const int concat_dimensions = output_shape.DimensionsCount();
  TFLITE_DCHECK_LT(axis, concat_dimensions);

  int64_t concat_size = 0;
  for (int i = 0; i < inputs_count; i++) {
    TFLITE_DCHECK_EQ(input_shapes[i]->DimensionsCount(), concat_dimensions);
    for (int j = 0; j < concat_dimensions; j++) {
      if (j != axis) {
        MatchingDim(*input_shapes[i], j, output_shape, j);
      }
    }
    concat_size += input_shapes[i]->Dims(axis);
  }
  TFLITE_DCHECK_EQ(concat_size, output_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= output_shape.Dims(i);
  }
  // For all input arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < concat_dimensions; ++i) {
    base_inner_size *= output_shape.Dims(i);
  }

  Scalar* output_ptr = output_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      const int copy_size = input_shapes[i]->Dims(axis) * base_inner_size;
      for (unsigned int idx = 0; idx < copy_size; idx++) {
        output_ptr[idx] = (input_data[i] + k * copy_size)[idx];
      }
      output_ptr += copy_size;
    }
  }
}

// TODO(prabhumk): This is the same as the optimized implementation.
// TODO(prabhumk): The quantized implementation of concatentation isn't fully
// quantized as it takes scale as a floating point value. This should be fixed
// when optimizng this routine further.
inline void ConcatenationWithScaling(const ConcatenationParams& params,
                                     const RuntimeShape* const* input_shapes,
                                     const uint8* const* input_data,
                                     const RuntimeShape& output_shape,
                                     uint8* output_data) {
  int axis = params.axis;
  const int32* input_zeropoint = params.input_zeropoint;
  const float* input_scale = params.input_scale;
  int inputs_count = params.inputs_count;
  const int32 output_zeropoint = params.output_zeropoint;
  const float output_scale = params.output_scale;

  const int concat_dimensions = output_shape.DimensionsCount();
  TFLITE_DCHECK_LT(axis, concat_dimensions);

  int64_t concat_size = 0;
  for (int i = 0; i < inputs_count; i++) {
    TFLITE_DCHECK_EQ(input_shapes[i]->DimensionsCount(), concat_dimensions);
    for (int j = 0; j < concat_dimensions; j++) {
      if (j != axis) {
        MatchingDim(*input_shapes[i], j, output_shape, j);
      }
    }
    concat_size += input_shapes[i]->Dims(axis);
  }
  TFLITE_DCHECK_EQ(concat_size, output_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= output_shape.Dims(i);
  }
  // For all input arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < concat_dimensions; ++i) {
    base_inner_size *= output_shape.Dims(i);
  }

  const float inverse_output_scale = 1.f / output_scale;
  uint8* output_ptr = output_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      const int copy_size = input_shapes[i]->Dims(axis) * base_inner_size;
      const uint8* input_ptr = input_data[i] + k * copy_size;
      if (input_zeropoint[i] == output_zeropoint &&
          input_scale[i] == output_scale) {
        for(int idx = 0; idx < copy_size; idx++) {
          output_ptr[idx] = input_ptr[idx];
        }
      } else {
        const float scale = input_scale[i] * inverse_output_scale;
        const float bias = -input_zeropoint[i] * scale;
        for (int j = 0; j < copy_size; ++j) {
          const int32_t value =
                  static_cast<int32_t>(std::round(input_ptr[j] * scale + bias)) +
                  output_zeropoint;
          output_ptr[j] =
                  static_cast<uint8_t>(std::max(std::min(255, value), 0));
        }
      }
      output_ptr += copy_size;
    }
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);

  int axis = params->axis;
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  if (axis < 0) axis += output->dims->size;

// TODO(ahentz): Creating 'all_inputs' below is not very efficient. We should
// allocate and populate these during Prepare().
// TODO(ycling): Activation function parameter is ignored. For now we dont have
// a model with a Concatenation with fused activation function.
#define TF_LITE_CONCATENATION(scalar)                                         \
  {                                                                           \
    VectorOfTensors<scalar> all_inputs(*context, *node->inputs);              \
    tflite::ConcatenationParams op_params;                                    \
    op_params.axis = axis;                                                    \
    op_params.inputs_count = node->inputs->size;                              \
    Concatenation(op_params, all_inputs.shapes(),                             \
                                 all_inputs.data(), GetTensorShape(output),   \
                                 GetTensorData<scalar>(output));              \
  }

#define TF_LITE_CONCATENATION_QUANTIZED()                         \
  {                                                               \
    VectorOfQuantizedTensors all_inputs(*context, *node->inputs); \
    tflite::ConcatenationParams op_params;                        \
    op_params.axis = axis;                                        \
    op_params.input_zeropoint = all_inputs.zero_point();          \
    op_params.input_scale = all_inputs.scale();                   \
    op_params.inputs_count = node->inputs->size;                  \
    op_params.output_zeropoint = output->params.zero_point;       \
    op_params.output_scale = output->params.scale;                \
    ConcatenationWithScaling(                                     \
        op_params, all_inputs.shapes(), all_inputs.data(),        \
        GetTensorShape(output), GetTensorData<uint8>(output));    \
  }

  switch (output->type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      TF_LITE_CONCATENATION(float);
      break;
    case kTfLiteInt32:
      TF_LITE_CONCATENATION(int32);
      break;
    case kTfLiteUInt8:
      TF_LITE_CONCATENATION_QUANTIZED();
      break;
    case kTfLiteInt8:
      TF_LITE_CONCATENATION(int8_t);
      break;
    case kTfLiteInt64:
      TF_LITE_CONCATENATION(int64_t);
      break;

    default:
      context->ReportError(context, "Type '%s' is not supported currently.",
                           TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

#undef TF_LITE_CONCATENATION_QUANTIZED
#undef TF_LITE_CONCATENATION

  return kTfLiteOk;
}

//#undef TF_LITE_MACRO_DISPATCH

}  // namespace concatenation

TfLiteRegistration* Register_CONCATENATION() {
  static TfLiteRegistration r = {concatenation::Init, concatenation::Free,
                                 concatenation::Prepare, concatenation::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
