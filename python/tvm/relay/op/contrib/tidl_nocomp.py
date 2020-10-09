# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
"""TIDL library supported operators.
"""
from ... import op as reg
from topi.util import get_const_tuple
from tvm import relay
from tvm.relay.frontend.common import infer_shape
from tvm.relay.frontend.common import infer_type

target = "target.tidl"

def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by TIDL.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """
    @reg.register(op_name, "target.tidl")
    def _func_wrapper(attrs, args):
        return supported
    return _func_wrapper

@reg.register("add", "target.tidl")
def _add_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("nn.argmax", "target.tidl")
def _argmax_whitelist_fn(attrs, args):
    keepdims = attrs.keepdims
    exclude = attrs.exclude
    axis = attrs.axis
    # is checked_type.shape always guaranteed to be there?
    data = args[0]
    supported = (int(infer_shape(data)[1]) <= 15 and keepdims == 1 and axis == 1 and exclude == 0)
    return supported

@reg.register("nn.avg_pool2d", "target.tidl")
def _avg_pool_whitelist_fn(attrs, args):
    pool_size = get_const_tuple(attrs.pool_size)
    strides = get_const_tuple(attrs.strides)
    supported = (pool_size[0] <= 9 and pool_size[1] <= 9 and strides[0] <= 3 and strides[1] <=2)
    return supported

@reg.register("nn.batch_flatten", "target.tidl")
def _batch_flatten_fn(attrs, args):
    data = args[0]
    if(len(infer_shape(data)) == 4):
        supported = (int(infer_shape(data)[2]) <= 65535 and int(infer_shape(data)[3]) <= 65535)
    else:
        supported = True
    return supported

@reg.register("nn.batch_norm", "target.tidl")
def _batch_norm_whitelist_fn(attrs, args):
    #These are the relay arguments... look up the operator to get the actual name...
    data0 = args[0]
    data1 = args[1]
    supported = True

    """
    if data1.checked_type.dtype != 'float32':
        supported = False
    elif attrs.data_layout == 'NCHW' and call_node.attrs.axis != 1:
        #only axis along channel is supported
        #attributes include parameters that are optional and having default values in operator arguments
        supported = False
    elif attrs.data_layout == 'NHWC' and attrs.axis != 3:
        supported = False
    """

    return supported

@reg.register("nn.bias_add", "target.tidl")
def _bias_add_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("clip", "target.tidl")
def _clip_whitelist_fn(attrs, args):
    a_min = attrs.a_min
    a_max = attrs.a_max
    supported = (a_min == 0 and a_max == 6)
    return supported

@reg.register("nn.concatenate", "target.tidl")
def _concatenate_whitelist_fn(attrs, args):
    supported = (attrs.axis == 1)
    return supported

@reg.register("nn.conv2d", "target.tidl")
def _conv2d_whitelist_fn(attrs, args):
    weight = args[1]
    if infer_type(weight) != 'float32':
        return False

    weight_shape  = get_const_tuple(infer_shape(weight))
    strides       = get_const_tuple(attrs.strides)
    dilation      = get_const_tuple(attrs.dilation)
    kernel_size   = get_const_tuple(attrs.kernel_size)

    (dh, dw) = dilation
    (kh, kw) = kernel_size
    channel_supported  = (weight_shape[0] <= 2048 and weight_shape[1] <= 2048)
    stride_supported   = (strides[0] <= 2 and strides[1] <= 2)
    dilation_supported = (dh == 1 or dh == 2 or dh == 4) and (dw == 1 or dw == 2 or dw == 4)
    kernel_supported = (((kh-1)*dh+1) <= 9) and (((kw-1)*dw+1) <= 9)
    supported = channel_supported and stride_supported and dilation_supported and kernel_supported
    return supported

@reg.register("nn.conv2d_transpose", "target.tidl")
def _conv2d_transpose_whitelist_fn(attrs, args):
    weight = args[1]
    weight_shape  = get_const_tuple(infer_shape(weight))
    strides = get_const_tuple(attrs.strides)
    groups = attrs.groups
    supported = (weight_shape[0] == weight_shape[1]) and (weight_shape[0] == groups) and (strides[1] == 2)
    return supported

@reg.register("nn.dense", "target.tidl")
def _dense_whitelist_fn(attrs, args):
    weight = args[1]

    #weight_shape = get_const_tuple(weight.checked_type.shape)
    weight_shape = infer_shape(weight)
    w_in  = weight_shape[1]
    w_out = weight_shape[0]
    supported = (w_in <= 65536) and (w_out <= 16384) and (w_in * w_out <= 67108864)
    return supported

@reg.register("nn.dropout", "target.tidl")
def _dropout_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("nn.global_avg_pool2d", "target.tidl")
def _global_avg_pool_whitelist_fn(attrs, args):
    data = args[1]
    data_shape  = get_const_tuple(infer_shape(data))
    layout = attrs.layout
    if layout == "NCHW":
        height = data_shape[2]
        width  = data_shape[3]
    else:
        height = data_shape[1]
        width  = data_shape[2]
    supported = (height * width <= 4096)
    return supported

@reg.register("nn.max_pool2d", "target.tidl")
def _max_pool_whitelist_fn(attrs, args):
    pool_size = get_const_tuple(attrs.pool_size)
    strides   = get_const_tuple(attrs.strides)
    supported = (pool_size[0] <= 9) and (pool_size[1] <= 9) and (strides[0] <= 3) and (strides[1] <= 2)
    return supported

@reg.register("multiply", "target.tidl")
def _multiply_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("nn.nms", "target.tidl")
def _nms_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("nn.pad", "target.tidl")
def _pad_whitelist_fn(attrs, args):
    supported = (call_node.attrs.pad_value == 0.0 and call_node.attrs.pad_mode == 'constant')
    return supported

@reg.register("nn.prelu", "target.tidl")
def _prelu_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("nn.relu", "target.tidl")
def _relu_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("nn.slice_like", "target.tidl")
def _slice_like_whitelist_fn(attrs, args):
    supported = (attrs.axis == 1)
    return supported

@reg.register("nn.softmax", "target.tidl")
def _softmax_whitelist_fn(attrs, args):
    supported = (attrs.axis != 2)
    return supported

@reg.register("split", "target.tidl")
def _split_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("squeeze", "target.tidl")
def _split_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("reshape", "target.tidl")
def _split_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("transpose", "target.tidl")
def _split_whitelist_fn(attrs, args):
    supported = True
    return supported
