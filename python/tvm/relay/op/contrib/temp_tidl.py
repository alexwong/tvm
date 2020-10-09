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
from tvm.relay import op as reg
from topi.util import get_const_tuple
from tvm import relay

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

"""

#Fuse get_valid_count and non_max_suppression into a single composite
#function which can be partitioned and converted to TRT.
def make_nms_pattern():
    x = relay.var('x')w
    ret = relay.vision.get_valid_counts(x, score_threshold=0)
    nms_out = relay.vision.non_max_suppression(ret[1], ret[0])
    return nms_out

"""

def _merge_sequential_ops(mod):
    """Fuse sequential ops for op registration. Ops: vision.multibox_prior, nn.reshape, squeeze, transpose
    """
    def _multibox_prior_pattern():
        #multibox_prior has to be followed by nn.concatenate or vision.nms
        return multibox_prior_out

    #reshape has to be preced by nn.avg_pool2d, nn.global_avg_pool2d, nn.dense, squeeze, transpose (if transpose, special path)
    #reshape should be followed by softmax or transpose (special path, has to be transpose, reshape, transpose)

    """
    patterns
    
    avgpool reshape 
    avgpool softmax
    
    avgpool globalavgpool reshape...
    basically a ton... write code to generate each pattern... needs to create a function
    """
    def _generate_reshape_patterns():
        input_ops = [relay.op.nn.avg_pool2d, relay.op.nn.global_avg_pool2d, relay.op.nn.dense, relay.op.transform.squeeze]
        output_ops = [relay.op.softmax]
        #wrapper_ops = []

        reshape_patterns = _generate_tidl_patterns(relay.op.transform.reshape, input_ops, output_ops)

        return reshape_patterns

    """
    takes set of possible input ops, output ops, and the op, and generates a list of functions
    also, ops that have to be used as wrappes
    
    what about inputs and stuff... and params to each function.. have to figure that out...
    
    have to deal with wrapper functions
    
    """
    def _generate_tidl_patterns(op, possible_inputs, possible_outputs):
        import itertools

        # Make below into a helper
        input_combinations = []
        for i in xrange(1, len(lst)+1):
            combinations = [list(x) for x in itertools.combinations(possible_inputs, i)]
            input_combinations.extend(combinations)

        output_combinations = []
        for i in xrange(1, len(lst)+1):
            combinations = [list(x) for x in itertools.combinations(possible_outputs, i)]
            output_combinations.extend(combinations)

        generated_functions = []
        for i in input_combinations:
            for o in output_combinations:
                pattern = i.append(op)
                pattern.append(o)
                generated_functions.extend(pattern)

        return generated_functions

    """
    def _reshape_pattern():
        return reshape_out
    """

    #squeese has to be followed by reshape
    def _squeeze_pattern():
        x = relay.var('x')
        y = relay.var('y')
        squeeze_out = relay.op.transform.squeeze(x)
        reshape_out = relay.op.transform.reshape(squeeze_out, y)
        return reshape_out

    #tranpose has to be followed by batch_flatten or reshape (special pattern)
    #transpose has to be preceded by reshape (special pattern, reshape transpose reshape)
    def _transpose_pattern():
        x = relay.var('x')
        transpose_out = relay.op.transform.transpose(x)
        batch_flatten_out = relay.op.nn.batch_flatten(transpose_out)
        return batch_flatten_out

    def _transpose_reshape_pattern():
        x = relay.var('x')
        y = relay.var('y')
        reshape_out1 = relay.op.transform.reshape(x, y)
        transpose_out = relay.op.transform.transpose(reshape_out1)
        reshape_out2 = relay.op.transform.reshape(transpose_out, y)
        return reshape_out2

    def _transpose_batch_reshape_pattern():
        x = relay.var('x')
        y = relay.var('y')
        reshape_out1 = relay.op.transform.reshape(x, y)
        transpose_out = relay.op.transform.transpose(reshape_out1)
        batch_flatten_out = relay.op.nn.batch_flatten(transpose_out)
        reshape_out2 = relay.op.transform.reshape(batch_flatten_out, y)
        return reshape_out2

    pattern_table = [
        ('tidl.multibox_prior', _multibox_prior_pattern()),
        ('tidl.reshape', _reshape_pattern()),
        ('tidl.squeeze', _squeeze_pattern()),
        ('tidl.transpose', _transpose_pattern())
        ('tidl.transpose_reshape', _transpose_reshape_pattern())
        ('tidl.tanspose_batch_reshape', _transpose_batch_reshape_pattern())
    ]

    return relay.transform.MergeComposite(pattern_table)(mod)

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
    supported = (int(data.checked_type.shape[1]) <= 15 and keepdims == 1 and axis == 1 and exclude == 0)
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
    if(len(data.checked_type.shape) == 4):
        supported = (int(data.checked_type.shape[2]) <= 65535 and int(data.checked_type.shape[3]) <= 65535)
    else:
        supported = True
    return supported

@reg.register("nn.batch_norm", "target.tidl")
def _batch_norm_whitelist_fn(attrs, args):
    #These are the relay arguments... look up the operator to get the actual name...
    data0 = args[0]
    data1 = args[1]
    supported = True

    if data1.checked_type.dtype != 'float32':
        supported = False
    elif attrs.data_layout == 'NCHW' and call_ndoe.attrs.axis != 1:
        #only axis along channel is supported
        #attributes include parameters that are optional and having default values in operator arguments
        supported = False
    elif attrs.data_layout == 'NHWC' and attrs.axis != 3:
        supported = False

    return supported

@reg.register("nn.bias_add", "target.tidl")
def _bias_add_whitelist_fn(attrs, args):
    supported = True
    return supported

@reg.register("nn.clip", "target.tidl")
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
    if weight.checked_type.dtype != 'float32':
        return False

    weight_shape  = get_const_tuple(weight.checked_type.shape)
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
    weight_shape  = get_const_tuple(weight.checked_type.shape)
    strides = get_const_tuple(attrs.strides)
    groups = attrs.groups
    supported = (weight_shape[0] == weight_shape[1]) and (weight_shape[0] == groups) and (strides[1] == 2)
    return supported

@reg.register("nn.dense", "target.tidl")
def _dense_whitelist_fn(attrs, args):
    weight = args[1]
    weight_shape = get_const_tuple(weight.checked_type.shape)
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
    data_shape  = get_const_tuple(data.checked_type.shape)
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

@reg.register("vision.multibox_prior", "target.tidl")
def _mutlibox_prior_whitelist_fn(attrs, args):
    supported = False
    # Use MergeCompositePass
    """
    supported = 0
    outCallNodes = find_out_call_nodes(node_dict, call_node)
    for idx in outCallNodes:
        if outCallNodes[idx].op.name == "nn.concatenate" or \
           outCallNodes[idx].op.name == "vision.nms":
            supported = 1

    return (supported)
    """
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

@reg.register("nn.reshape", "target.tidl")
def _rehsape_whitelist_fn(attrs, args):
    supported = False
    # Use MergeCompositePass
    """
    elif call_node.op.name == "reshape":
        supported = False
        reshape_after_transpose = False
        transpose_after_reshape = False
        inpCallNodes = find_in_call_nodes(node_dict, call_node)
        for idx in inpCallNodes:
            if inpCallNodes[idx].op.name == "nn.avg_pool2d" or \
               inpCallNodes[idx].op.name == "nn.global_avg_pool2d" or \
               inpCallNodes[idx].op.name == "nn.dense" or \
               inpCallNodes[idx].op.name == "squeeze":
                supported = True
            elif inpCallNodes[idx].op.name == "transpose":
                reshape_after_transpose = True

        outCallNodes = find_out_call_nodes(node_dict, call_node)
        for idx in outCallNodes:
            if outCallNodes[idx].op.name == "nn.softmax":
                supported = True
            elif outCallNodes[idx].op.name == "transpose":
                transpose_after_reshape = True

        if reshape_after_transpose and transpose_after_reshape:
            supported = True

        # If this is the last node of the graph, and input and output shape are 
        # the same, this operator can be supported by TIDL
        if len(outCallNodes) ==0:
            node_is_identity = True
            for idx in range(len(data.checked_type.shape)):
                if int(data.checked_type.shape[idx]) != int(call_node.attrs.newshape[idx]):
                    node_is_identity = False
            if node_is_identity == True:
                supported = True

        return (supported)
    """
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
def _squeeze_whitelist_fn(attrs, args):
    supported = False
    # Use MergeCompositePass
    """
    outCallNodes = find_out_call_nodes(node_dict, call_node)
    for idx in outCallNodes:
        if outCallNodes[idx].op.name == "reshape":
            supported = True
    """
    return supported

@reg.register("transpose", "target.tidl")
def _transpose_whitelist_fn(attrs, args):
    supported = False
    # Use MergeCompositePass
    """
    supported = False
    reshape_after_transpose = False
    transpose_after_reshape = False
    outCallNodes = find_out_call_nodes(node_dict, call_node)
    for idx in outCallNodes:
        if outCallNodes[idx].op.name == "nn.batch_flatten":
            supported = True
        elif outCallNodes[idx].op.name == "reshape":
            reshape_after_transpose = True

    inpCallNodes = find_in_call_nodes(node_dict, call_node)
    for idx in inpCallNodes:
        if inpCallNodes[idx].op.name == "reshape":
            transpose_after_reshape = True

    if reshape_after_transpose and transpose_after_reshape:
        supported = True
    """
    return supported

"""
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.relu")
_register_external_op_helper("add")
_register_external_op_helper("subtract")
_register_external_op_helper("multiply")
"""

def _generate_reshape_op_patterns():
    input_ops = ["relay.op.nn.avg_pool2d", "relay.op.nn.global_avg_pool2d", "relay.op.nn.dense", "relay.op.transform.squeeze", "relay.op.transform.transpose"]
    output_ops = ["relay.op.nn.softmax", "relay.op.transform.transpose"]
    #wrapper_ops = []

    reshape_patterns = _generate_tidl_patterns("relay.op.transform.reshape", input_ops, output_ops)
    print("rehsape patterns")
    print(reshape_patterns)
    print(len(reshape_patterns))


    print("iterate reshape patterns")

    """
    i = 1
    for reshape_pattern in reshape_patterns[:]:
        if reshape_pattern.count("relay.op.transform.transpose") == 2:
            reshape_patterns.remove(reshape_pattern)
    """

    cnt = 1
    for reshape_pattern in reshape_patterns:
        print(cnt)
        print(reshape_pattern)
        cnt = cnt+1

    return reshape_patterns


"""
takes set of possible input ops, output ops, and the op, and generates a list of functions
also, ops that have to be used as wrappes

what about inputs and stuff... and params to each function.. have to figure that out...

have to deal with wrapper functions

"""
def _generate_tidl_patterns(op, possible_inputs, possible_outputs):
    import itertools

    # Make below into a helper
    input_combinations = []
    for i in range(1, len(possible_inputs)+1):
        combinations = [list(x) for x in itertools.combinations(possible_inputs, i)]
        permutations = itertools.permutations(possible_inputs)
        input_combinations.extend(permutations)

    output_combinations = []
    for i in range(1, len(possible_outputs)+1):
        combinations = [list(x) for x in itertools.combinations(possible_outputs, i)]
        permutations = itertools.permutations(possible_outputs)
        output_combinations.extend(permutations)

    generated_functions = []

    print("input combinations")
    print(input_combinations)
    print(len(input_combinations))

    print("output combinations")
    print(output_combinations)
    print(len(output_combinations))

    """
    for i in input_combinations:
        for o in output_combinations:
            print("input")
            print(i)

            print("output")
            print(o)

            print("op")
            print(op)

            pattern = []
            pattern = i
            pattern.append(op)
            pattern.append(o)

            print("pattern")
            print(pattern)

            generated_functions.extend([pattern])
            print("add a new generated pattern")
            print(len(generated_functions))

    return generated_functions
    """

    for inputc in input_combinations:
        print('inputc')
        print(inputc)
        for outputc in output_combinations:
            new_pattern = []
            print('outputc')
            print(outputc)
            for i in inputc:
                new_pattern.append(i)
            new_pattern.append(op)
            for o in outputc:
                new_pattern.append(o)
            print('adding a new pattern')
            print(new_pattern)
            generated_functions.append(new_pattern)

    #empty preamble
    for outputc in output_combinations:
        new_pattern = []
        new_pattern.append(op)
        for o in outputc:
            new_pattern.append(o)
        generated_functions.append(new_pattern)

    #empty postamble
    for inputc in input_combinations:
        new_pattern = []
        for i in inputc:
            new_pattern.append(i)
        new_pattern.append(op)
        generated_functions.append(new_pattern)

    for generated_fn in generated_functions[:]:
        if (generated_fn.count("relay.op.transform.transpose") ==1):
            print('remove this')
            print(generated_fn)
            generated_functions.remove(generated_fn)

    cnt = 0
    print('get generated_fns')
    for generated_fn in generated_functions:
        print(cnt)
        print(generated_fn)
        cnt = cnt+1

    return generated_functions

_generate_reshape_op_patterns()
