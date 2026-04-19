#include "op/layer.h"
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>
#include "base/cuda_config.h"

namespace op {
// BaseLayer
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                     std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {}

base::DataType BaseLayer::data_type() const { return data_type_; }

LayerType BaseLayer::layer_type() const { return layer_type_; }

base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
  return base::error::FunctionNotImplement();
}

base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                   const void* weight_ptr, base::DeviceType device_type) {
  return base::error::FunctionNotImplement();
}

const std::string& BaseLayer::get_layer_name() const { return layer_name_; }

void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }
base::DeviceType BaseLayer::device_type() const { return device_type_; }

void BaseLayer::set_device_type(base::DeviceType device_type) { device_type_ = device_type; }

// Layer
Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {}

base::Status Layer::init() { return base::error::Success(); }

base::Status Layer::forward() { return base::error::FunctionNotImplement(""); }

base::Status Layer::check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                                 base::DataType data_type) const {
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }
  return base::error::Success();
}

base::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor,
                                          base::DeviceType device_type, base::DataType data_type,
                                          std::initializer_list<int32_t> dims) const {
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }

  // 先检查张量维度数是否相等
  if (tensor.dims_size() != static_cast<int32_t>(dims.size())) {
    return base::error::InvalidArgument(
        "Tensor dimension count does not match the supplied dims count");
  }

  // 逐维度比较
  for (auto i = 0; i < dims.size(); ++i) {
    if (tensor.get_dim(i) != *(dims.begin() + i)) {
      return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
    }
  }
  return base::error::Success();
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_[idx] = input;
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_[idx] = output;
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_[idx];
}

tensor::Tensor& Layer::get_input(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_[idx];
}

tensor::Tensor& Layer::get_output(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_[idx];
}

base::Status Layer::check() const {
  return base::error::FunctionNotImplement("The check function is not implement yet");
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_[idx];
}

void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

void Layer::reset_output_size(size_t size) { outputs_.resize(size); }

void Layer::to_cuda() {
  for (auto& input : inputs_) {
    if (!input.is_empty()) {
      input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }

  for (auto& output : outputs_) {
    if (!output.is_empty()) {
      output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
  if (!config) {
    return;
  }
  this->cuda_config_ = config;
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const { return cuda_config_; }

size_t Layer::input_size() const { return inputs_.size(); }

size_t Layer::output_size() const { return outputs_.size(); }

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& input5, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);

  this->set_output(0, output1);
  return this->forward();
}

// LayerParam
LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer,
                       std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)), is_quant_layer_(is_quant_layer) {}

base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
  if (!weight.is_empty()) {
    CHECK(weight.device_type() == device_type_);
  }
  weights_[idx] = weight;
  return base::error::Success();
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_[idx];
}

void LayerParam::to_cuda() {
  Layer::to_cuda();
  for (auto& weight : weights_) {
    weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
  if (!scales_.is_empty()) {
    scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
}

base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    /*权重内存指针*/ const void* weight_ptr,
                                    base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK_NE(weight_ptr, nullptr);

  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
  // 创建共享buffer
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  // 非量化层只设置weight
  if (!is_quant_layer_) {
    tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;
  } else {
    // is quant layer
    tensor::Tensor weight(base::DataType::kDataTypeInt8, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;

    const int32_t weight_size = static_cast<int32_t>(weight.size());
    CHECK(weight_size % group_size_ == 0);

    /*
    背景：在模型文件（如 ggml、QNN 等）里，每 group_size_ 个 INT8 权重会共享一个 FP32
    scale。文件布局通常是：
    | int8 weight block (N bytes) | float scale block (N/group_size_ * sizeof(float)) |
    */
    int32_t scale_nums = weight_size / group_size_;
    scales_ = tensor::Tensor{
        base::DataType::kDataTypeFp32, scale_nums, false,  // 不拥有当前内存
        nullptr,
        reinterpret_cast<float*>(
            (int8_t*)weight_ptr +
            weight_size)  // 把指针移动到weight_size字节后的位置(参考上面的布局),纸箱scale区域的起始
    };
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

void LayerParam::set_scales(const tensor::Tensor& scales) {
  CHECK(!scales.is_empty());
  this->scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size) { this->group_size_ = group_size; }

int32_t LayerParam::get_scale_num() const {
  CHECK(!scales_.is_empty());
  return static_cast<int32_t>(scales_.size());
}

void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

size_t LayerParam::weight_size() const { return weights_.size(); }

tensor::Tensor& LayerParam::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}
}  // namespace op