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

/*!
 * \file precompute_prune.cc
 * \brief Split the graph into a pre-compute graph and a execution graph.
 *
 *  The pre-compute graph outputs parameters that can be taken
 *  by execution graph during execution phase.
 */
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/compiler/op_attr_types.h>
#include <unordered_set>

namespace nnvm {
namespace compiler {

nnvm::Graph PrecomputePrune(nnvm::Graph src) {
  const auto& plist
      = src.GetAttr<std::vector<std::string> >("param_name_list");
  std::unordered_set<std::string> params(plist.begin(), plist.end());

  std::unordered_set<nnvm::Node*> pruned;
  nnvm::NodeEntryMap<nnvm::NodePtr> entry_var;
  std::unordered_set<std::string> unique_name;
  // number of edges that are not variable
  int non_var_edge = 0;
  std::unordered_map<Node*, std::vector<NodeEntry> > version_hist;

  auto replace_pruned_entry = [&] (const NodeEntry& e) {
    if (!entry_var.count(e)) {
      if (!e.node->is_variable()) {
        ++non_var_edge;
      }
      nnvm::NodePtr var = nnvm::Node::Create();
      var->attrs.name = e.node->attrs.name;
      if (e.version && version_hist.count(e.node.get()) == 0) {
        var->attrs.name += "_" + std::to_string(e.version);
        version_hist[e.node.get()] = std::vector<NodeEntry>{};
      }
      if (e.node->num_outputs() != 1) {
        var->attrs.name += "_output" + std::to_string(e.index);
      }
      entry_var.emplace(e, var);
      CHECK(!unique_name.count(var->attrs.name));
      unique_name.insert(var->attrs.name);
      return nnvm::NodeEntry{var, 0, 0};
    } else {
      return nnvm::NodeEntry{entry_var.at(e), 0, 0};
    }
  };

  DFSVisit(src.outputs, [&](const nnvm::NodePtr& n) {
    bool can_be_pruned = true;
    if (n->is_variable()) {
      if (params.count(n->attrs.name)) {
        pruned.emplace(n.get());
      }
      can_be_pruned = false;
    }

    for (const auto& e : n->inputs) {
      if (!pruned.count(e.node.get())) {
        can_be_pruned = false;
      }
    }
    if (can_be_pruned) {
      pruned.emplace(n.get());
    } else {
      // scan again to find edge nodes, skip variables
      for (auto& e : n->inputs) {
        if (pruned.count(e.node.get())) {
          e = replace_pruned_entry(e);
        }
      }
    }
  });

  // nothing being pruned.
  if (non_var_edge == 0 && version_hist.size() == 0) {
    return src;
  }

  for (auto& e : src.outputs) {
    if (pruned.count(e.node.get())) {
      e = replace_pruned_entry(e);
    }
  }

  nnvm::Graph pre_graph;
  pre_graph.outputs.reserve(entry_var.size());
  std::vector<std::string> output_names;
  output_names.reserve(entry_var.size());

  for (auto kv : entry_var) {
    pre_graph.outputs.emplace_back(kv.first);
    output_names.emplace_back(kv.second->attrs.name);
  }
  // new parameter list
  pre_graph.attrs["output_names"] =
      std::make_shared<dmlc::any>(std::move(output_names));
  src.attrs["precompute_graph"] =
      std::make_shared<dmlc::any>(std::move(pre_graph));
  return src;
}

NNVM_REGISTER_PASS(PrecomputePrune)
.set_body(PrecomputePrune);
}  // namespace compiler
}  // namespace nnvm