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

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_BN_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_BN_H_

#if MXNET_USE_MKLDNN == 1
#include "../common.h"
#include "../subgraph_property.h"

namespace mxnet {
namespace op {

class SgMKLDNNBnSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    sFail = 0,
    sStart,
    sSuccess,
  };

 private:
  bool disable_bn_relu;
  SelectStatus status;
  std::vector<const nnvm::Node *> matched_list;

 public:
  SgMKLDNNBnSelector(int dis_bn_relu): disable_bn_relu(dis_bn_relu){}

  bool Select(const nnvm::Node &n) override {
    bool match =
        (!disable_bn_relu) && (!n.is_variable()) && (n.op()->name == "BatchNorm");
    if (match) {
      status = sStart;
      matched_list.clear();
      matched_list.push_back(&n);
      return true;
    }
    return false;
  }

  bool SelectInput(const nnvm::Node &n,
                           const nnvm::Node &new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node &n,
                            const nnvm::Node &new_node) override {
    if (new_node.is_variable()){
      return false;
    }

    // Use status machine to do selection.
    switch (status) {
      case sStart:
        if ((!disable_bn_relu) && new_node.op()->name == "Activation") {
          status = sSuccess;
          return true;
        } else {
          status = sFail;
          return false;
        }
      case sFail:
        return false;
      case sSuccess:
        if (matched_list.back() == &n) {
          status = sFail;
          return false;
        }
        return false;
      default:
        return false;
    }
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (status == sFail) {
      return std::vector<nnvm::Node *>(0);
    } else {
      return candidates;
    }
  }
};

class SgMKLDNNBnProperty : public SubgraphProperty {
 public:
  SgMKLDNNBnProperty() {
    int disable_all = dmlc::GetEnv("MXNET_DISABLE_FUSION_ALL", 0);
    disable_bn_relu = dmlc::GetEnv("MXNET_DISABLE_FUSION_BN_RELU", 0);

    if (disable_all || disable_bn_relu) {
      LOG(INFO) << "MKLDNN BatchNormalization fusion pass is disabled.";
    } else {
      LOG(INFO) << "Start to execute MKLDNN BatchNormalization fusion pass.";
    }

    if (disable_all) {
      disable_bn_relu = 1;
    }
  }
  
  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgMKLDNNBnProperty>();
  }

  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::NodePtr n = nnvm::Node::Create();

    // Initialize new attributes to false
    n->attrs.dict["with_relu"] = "false";

    // This op has single output, remove duplicated.
    auto last_node = sym.outputs[0].node;
    nnvm::Symbol new_sym;
    new_sym.outputs.emplace_back(nnvm::NodeEntry{last_node, 0, 0});
    std::string node_name = "";
    DFSVisit(new_sym.outputs, [&](const nnvm::NodePtr &node) {
      if (node->is_variable()) return;
      auto &sub_name = node->op()->name;
      if (sub_name == "BatchNorm") {
        node_name += "bn_";
      } else if (sub_name == "Activation") {
        node_name += "relu_";
        n->attrs.dict["with_relu"] = "true";
      }
    });

    n->attrs.name = "sg_mkldnn_" + node_name + std::to_string(subgraph_id);
    n->attrs.op = Op::Get("_sg_mkldnn_bn");
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
    n->attrs.parsed = new_sym;
    return n;
  }

  virtual SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgMKLDNNBnSelector>(
        disable_bn_relu);
    return selector;
  }

  void ConnectSubgraphOutput(
      const nnvm::NodePtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      *output_entries->at(i) = nnvm::NodeEntry{n, 0, 0};
    }
  }

  void ConnectSubgraphInput(
      const nnvm::NodePtr n, std::vector<nnvm::NodeEntry *> *input_entries,
      std::vector<nnvm::NodeEntry> *orig_input_entries) const override {
  }

 private:
  int disable_bn_relu;
};

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN, SgMKLDNNBnProperty);

} // namespace op
} // namespace mxnet
#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_BN_H_
