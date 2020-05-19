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

# numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys

# tvm, relay
import tvm
from tvm import te
from tvm import relay
from ctypes import *
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet

from tvm.relay.testing.darknet import LAYERTYPE
from tvm.relay.frontend.darknet import ACTIVATION
from tvm.contrib import graph_runtime
import time

perf = {'dn': 0.0, 'tvm': 0.0}
perf_dict = {}

#Helper
def _read_memory_buffer(shape, data, dtype='float32'):
    length = 1
    for x in shape:
        length *= x
    data_np = np.zeros(length, dtype=dtype)
    for i in range(length):
        data_np[i] = data[i]
    return data_np.reshape(shape)

def _get_tvm_output(net, data, build_dtype='float32', states=None):
    '''Compute TVM output'''
    dtype = 'float32'
    mod, params = relay.frontend.from_darknet(net, data.shape, dtype)
    target = 'llvm'
    shape_dict = {'data': data.shape}
    graph, library, params = relay.build(mod,
                                         target,
                                         params=params)

    # Execute on TVM
    ctx = tvm.cpu(0)
    m = graph_runtime.create(graph, library, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(data.astype(dtype)))
    if states:
        for name in states.keys():
            m.set_input(name, tvm.nd.array(states[name].astype(dtype)))
    m.set_input(**params)
    tvm_start = time.time()
    m.run()
    perf['tvm'] = (time.time() - tvm_start)
    # get outputs
    tvm_out = []
    for i in range(m.get_num_outputs()):
        tvm_out.append(m.get_output(i).asnumpy())
    return tvm_out

def _load_net(cfg_url, cfg_name, weights_url, weights_name):
    cfg_path = download_testdata(cfg_url, cfg_name, module='darknet')
    weights_path = download_testdata(weights_url, weights_name, module='darknet')
    net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
    return net

def verify_darknet_frontend(net, img_path, build_dtype='float32'):
    '''Test network with given input image on both darknet and tvm'''
    def get_darknet_output(net, img):
        dn_start = time.time()
        DARKNET_LIB.network_predict_image(net, img)
        perf['dn'] = (time.time() - dn_start)
        out = []
        for i in range(net.n):
            layer = net.layers[i]
            if layer.type == LAYERTYPE.REGION:
                attributes = np.array([layer.n, layer.out_c, layer.out_h,
                                       layer.out_w, layer.classes,
                                       layer.coords, layer.background],
                                      dtype=np.int32)
                out.insert(0, attributes)
                out.insert(0, _read_memory_buffer((layer.n*2, ), layer.biases))
                layer_outshape = (layer.batch, layer.out_c,
                                  layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(layer_outshape, layer.output))
            elif layer.type == LAYERTYPE.YOLO:
                attributes = np.array([layer.n, layer.out_c, layer.out_h,
                                       layer.out_w, layer.classes,
                                       layer.total],
                                      dtype=np.int32)
                out.insert(0, attributes)
                out.insert(0, _read_memory_buffer((layer.total*2, ), layer.biases))
                out.insert(0, _read_memory_buffer((layer.n, ), layer.mask, dtype='int32'))
                layer_outshape = (layer.batch, layer.out_c,
                                  layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(layer_outshape, layer.output))
            elif i == net.n-1:
                if layer.type == LAYERTYPE.CONNECTED:
                    darknet_outshape = (layer.batch, layer.out_c)
                elif layer.type in [LAYERTYPE.SOFTMAX]:
                    darknet_outshape = (layer.batch, layer.outputs)
                else:
                    darknet_outshape = (layer.batch, layer.out_c,
                                        layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(darknet_outshape, layer.output))
        return out

    dtype = 'float32'

    img = DARKNET_LIB.letterbox_image(DARKNET_LIB.load_image_color(img_path.encode('utf-8'), 0, 0), net.w, net.h)

    """
    import time
    start = time.time()
    for _ in range(1000):
        response = predictor.predict(payload)
    neo_inference_time = (time.time()-start)
    """

    #import time
    #dn_start = time.time()
    darknet_output = get_darknet_output(net, img)
    #dn_inference_time = (time.time()-dn_start)

    batch_size = 1
    data = np.empty([batch_size, img.c, img.h, img.w], dtype)
    i = 0
    for c in range(img.c):
        for h in range(img.h):
            for k in range(img.w):
                data[0][c][h][k] = img.data[i]
                i = i + 1

    #tvm_start = time.time()
    tvm_out = _get_tvm_output(net, data, build_dtype)
    #tvm_inference_time = (time.time()-tvm_start)

    for tvm_outs, darknet_out in zip(tvm_out, darknet_output):
        tvm.testing.assert_allclose(darknet_out, tvm_outs, rtol=1e-3, atol=1e-3)

    print('dn perf')
    print(perf['dn'])
    print('tvm perf')
    print(perf['tvm'])
    print('speedup')
    print(perf['dn']/perf['tvm'])
    perf_dict[img_path.split('/')[-1]] = perf['dn']/perf['tvm']

######################################################################
# Choose the model
# -----------------------
# Models are: 'yolov2', 'yolov3' or 'yolov3-tiny'

# Model name
MODEL_NAME = 'yolov2'

######################################################################
# Download required files
# -----------------------
# Download cfg and weights file if first time.
CFG_NAME = 'Body.cfg'
WEIGHTS_NAME = 'Body.weights'
REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'
DARKNET_TEST_IMAGE_PATH = '/Users/wongale/workplace/neo-tvm-pytorch-relay/tvm/validation/000000001518.jpg'
DARKNET_TEST_IMAGE_FOLDER = '/Users/wongale/workplace/neo-tvm-pytorch-relay/tvm/validation/'
cfg_path = CFG_NAME
weights_path = WEIGHTS_NAME

# Download and Load darknet library
if sys.platform in ['linux', 'linux2']:
    DARKNET_LIB = 'libdarknet2.0.so'
    DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
elif sys.platform == 'darwin':
    DARKNET_LIB = 'libdarknet_mac2.0.so'
    DARKNET_URL = REPO_URL + 'lib_osx/' + DARKNET_LIB + '?raw=true'
else:
    err = "Darknet lib is not supported on {} platform".format(sys.platform)
    raise NotImplementedError(err)

lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)

build_dtype = {}
import os
files = os.listdir('validation')
imgs = [x for x in files if '.jpg' in x]
imgs = imgs[0:100]
#imgs = ['000000080708.jpg', '000000164142.jpg', '000000249363.jpg', '000000332876.jpg', '000000417016.jpg', '000000500390.jpg', 'carsgraz_263.jpg', 'person_104.jpg']
for img in imgs:
    img_path = DARKNET_TEST_IMAGE_FOLDER+img
    verify_darknet_frontend(net, img_path, build_dtype)

print('perf_dict')
print(perf_dict)

perf_avg = sum(perf_dict.values()) / float(len(perf_dict))
print('perf_dict avg')
print(perf_avg)

#verify_darknet_frontend(net, DARKNET_TEST_IMAGE_PATH, build_dtype)

"""
dtype = 'float32'
batch_size = 1

data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape_dict = {'data': data.shape}
print("Converting darknet to relay functions...")
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

print(mod)
import json
json_str = tvm.ir.save_json(mod)
with open('darknet_yolo.json', 'w') as outfile:
    json.dump(json_str, outfile)

######################################################################
# Import the graph to Relay
# -------------------------
# compile the model
target = 'llvm'
target_host = 'llvm'
ctx = tvm.cpu(0)
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape = {'data': data.shape}
print("Compiling the model...")
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host=target_host,
                                     params=params)

print(shape)

[neth, netw] = shape['data'][2:] # Current image shape is 608x608
######################################################################
# Load a test image
# -----------------
"""
"""
test_image = 'dog.jpg'
print("Loading the test image...")
img_url = REPO_URL + 'data/' + test_image + '?raw=true'
img_path = download_testdata(img_url, test_image, "data")
"""
"""
img_path = '/Users/wongale/workplace/neo-tvm-pytorch-relay/tvm/validation/000000001518.jpg'
data = tvm.relay.testing.darknet.load_image(img_path, netw, neth)
######################################################################
# Execute on TVM Runtime
# ----------------------
# The process is no different from other examples.
from tvm.contrib import graph_runtime

m = graph_runtime.create(graph, lib, ctx)

# set inputs
m.set_input('data', tvm.nd.array(data.astype(dtype)))
m.set_input(**params)
# execute
print("Running the test image...")

# detection
# thresholds
thresh = 0.5
nms_thresh = 0.45

m.run()
# get outputs
tvm_out = []
"""
"""
if MODEL_NAME == 'yolov2':
    layer_out = {}
    layer_out['type'] = 'Region'
    # Get the region layer attributes (n, out_c, out_h, out_w, classes, coords, background)
    layer_attr = m.get_output(2).asnumpy()
    layer_out['biases'] = m.get_output(1).asnumpy()
    out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                 layer_attr[2], layer_attr[3])
    layer_out['output'] = m.get_output(0).asnumpy().reshape(out_shape)
    layer_out['classes'] = layer_attr[4]
    layer_out['coords'] = layer_attr[5]
    layer_out['background'] = layer_attr[6]
    tvm_out.append(layer_out)

elif MODEL_NAME == 'yolov3':
    for i in range(3):
        layer_out = {}
        layer_out['type'] = 'Yolo'
        # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
        layer_attr = m.get_output(i*4+3).asnumpy()
        layer_out['biases'] = m.get_output(i*4+2).asnumpy()
        layer_out['mask'] = m.get_output(i*4+1).asnumpy()
        out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                     layer_attr[2], layer_attr[3])
        layer_out['output'] = m.get_output(i*4).asnumpy().reshape(out_shape)
        layer_out['classes'] = layer_attr[4]
        tvm_out.append(layer_out)

elif MODEL_NAME == 'yolov3-tiny':
    for i in range(2):
        layer_out = {}
        layer_out['type'] = 'Yolo'
        # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
        layer_attr = m.get_output(i*4+3).asnumpy()
        layer_out['biases'] = m.get_output(i*4+2).asnumpy()
        layer_out['mask'] = m.get_output(i*4+1).asnumpy()
        out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                     layer_attr[2], layer_attr[3])
        layer_out['output'] = m.get_output(i*4).asnumpy().reshape(out_shape)
        layer_out['classes'] = layer_attr[4]
        tvm_out.append(layer_out)
        thresh = 0.560
"""

"""
# do the detection and bring up the bounding boxes
img = tvm.relay.testing.darknet.load_image_color(img_path)
_, im_h, im_w = img.shape
dets = tvm.relay.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), thresh,
                                                           1, tvm_out)
last_layer = net.layers[net.n - 1]
tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)

coco_name = 'coco.names'
coco_url = REPO_URL + 'data/' + coco_name + '?raw=true'
font_name = 'arial.ttf'
font_url = REPO_URL + 'data/' + font_name + '?raw=true'
coco_path = download_testdata(coco_url, coco_name, module='data')
font_path = download_testdata(font_url, font_name, module='data')

with open(coco_path) as f:
    content = f.readlines()

names = [x.strip() for x in content]

tvm.relay.testing.yolo_detection.draw_detections(font_path, img, dets, thresh, names, last_layer.classes)
plt.imshow(img.transpose(1, 2, 0))
plt.show()
"""
