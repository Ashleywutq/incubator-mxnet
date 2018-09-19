/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

#if MXNET_USE_MKLDNN == 1
#include <nnvm/graph.h>
#include <mshadow/base.h>
#include "../common.h"
#include "../../nn/batch_norm-inl.h"
#include "../../nn/activation-inl.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../nn/mkldnn/mkldnn_batch_norm-inl.h"
#include "../../../imperative/imperative_utils.h"
#include "../../../imperative/cached_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(BatchNormFusionParam);

static void BatchNormFusionFallBackCompute() {
  LOG(FATAL) << "Don't know how to do BnFusionFallBackCompute!";
}

static void BatchNormFusionComputeExCPU(const nnvm::NodeAttrs &bn_attrs,
                                          const OpContext &ctx,
                                          const std::vector<NDArray> &inputs,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<NDArray> &outputs) {

  CHECK_EQ(inputs.size(), 5U);
  const BatchNormParam &param = nnvm::get<BatchNormParam>(bn_attrs.parsed);
  // MKLDNN batchnorm only works well on the special MKLDNN layout.
  if (SupportMKLDNNBN(inputs[0], param) && inputs[0].IsMKLDNNData()) {
    std::vector<NDArray> in_data(inputs.begin(), inputs.begin() + batchnorm::kInMovingMean);
    std::vector<NDArray> aux_states(inputs.begin() + batchnorm::kInMovingMean, inputs.end());
    if (inputs[0].dtype() == mshadow::kFloat32) {
      //MKLDNN_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
      MKLDNNBatchNormForward<float>(ctx, bn_attrs, in_data, req, outputs, aux_states);
      //MKLDNN_OPCHECK_RUN(BatchNormCompute<cpu>, attrs, ctx, inputs, req, outputs);
      return;
    }
  }
  BatchNormFusionFallBackCompute();
}

class SgMKLDNNBnOperator {
 public:
  explicit SgMKLDNNBnOperator(const nnvm::NodeAttrs &attrs)
      : subgraph_sym_(nnvm::get<Symbol>(attrs.parsed)),
        bn_attrs_(nullptr),
        with_relu(false) {
    auto it = attrs.dict.find("with_relu");
    if (it != attrs.dict.end())
      with_relu = (it->second == "true");

    DFSVisit(subgraph_sym_.outputs, [&](const nnvm::NodePtr &node) {
      if (node->is_variable()) return;
      auto &node_name = node->op()->name;
      if (node_name == "BatchNorm") {
        CHECK(bn_attrs_.get() == nullptr);
        bn_attrs_ = std::make_shared<nnvm::NodeAttrs>(node->attrs);
      }
    });

    CHECK(bn_attrs_.get());
    bn_attrs_->dict["with_relu"] = with_relu ? "true" : "false";
  }

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx,
                const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn BatchNormalization only supports inference computation";
  }

 private:
  nnvm::Symbol subgraph_sym_;
  std::shared_ptr<nnvm::NodeAttrs> bn_attrs_;
  bool with_relu;
};

void SgMKLDNNBnOperator::Forward(const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs) {

    auto output = outputs[0];
    std::vector<NDArray> new_outputs={output};
    new_outputs.emplace_back(mxnet::kDefaultStorage, inputs[1].shape(),
                             inputs[1].ctx());
    new_outputs.emplace_back(mxnet::kDefaultStorage, inputs[2].shape(),
                             inputs[2].ctx());
    BatchNormFusionComputeExCPU(*bn_attrs_, ctx, inputs, req, new_outputs);
  }

  OpStatePtr CreateSgMKLDNNBnOpState(const nnvm::NodeAttrs &attrs, Context ctx,
                                       const std::vector<TShape> &in_shapes,
                                       const std::vector<int> &in_types) {
    return OpStatePtr::Create<SgMKLDNNBnOperator>(attrs);
  }

void SgMKLDNNBnOpForward(const OpStatePtr &state_ptr, const OpContext &ctx,
                             const std::vector<NDArray> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<NDArray> &outputs) {
    SgMKLDNNBnOperator &op = state_ptr.get_state<SgMKLDNNBnOperator>();
    op.Forward(ctx, inputs, req, outputs);
}

NNVM_REGISTER_OP(_sg_mkldnn_bn)
      .describe(R"code(_sg_mkldnn_bn)code" ADD_FILELINE)
      .set_num_inputs(DefaultSubgraphOpNumInputs)
      .set_num_outputs(DefaultSubgraphOpNumOutputs)
      .set_attr<nnvm::FListInputNames>("FListInputNames",
                                       DefaultSubgraphOpListInputs)
      .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                        [](const NodeAttrs &attrs) {
                                          return std::vector<std::string>{
                                              "output"};
                                        })
      .set_attr<FCreateOpState>("FCreateOpState", CreateSgMKLDNNBnOpState)
      .set_attr<nnvm::FInferShape>("FInferShape", DefaultSubgraphOpShape)
      .set_attr<nnvm::FInferType>("FInferType", DefaultSubgraphOpType)
      .set_attr<FInferStorageType>("FInferStorageType",
                                   DefaultSubgraphOpStorageType)
      .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>",
                                    SgMKLDNNBnOpForward)
      .set_attr<nnvm::FMutateInputs>("FMutateInputs",
                                     DefaultSubgraphOpMutableInputs)
      .set_attr<FResourceRequest>("FResourceRequest",
                                  DefaultSubgraphOpResourceRequest)
      .set_attr<std::string>("key_var_num_args", "num_args")
      .set_attr<bool>("TIsMKLDNN", true);
}  // namespace op
}  // namespace mxnet
#endif  // if MXNET_USE_MKLDNN == 1
