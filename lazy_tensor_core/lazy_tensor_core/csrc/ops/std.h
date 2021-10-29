#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Std : public TsNode {
 public:
  Std(const torch::lazy::Value& input, std::vector<int64_t> dimensions,
      bool keep_reduced_dimensions, int64_t correction);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<int64_t>& dimensions() const { return dimensions_; }

  bool keep_reduced_dimensions() const { return keep_reduced_dimensions_; }

  int64_t correction() const { return correction_; }

 private:
  std::vector<int64_t> dimensions_;
  bool keep_reduced_dimensions_;
  int64_t correction_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors