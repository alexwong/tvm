import torch
from torchvision.ops import roi_align, nms
import tvm

from tvm import relay
from tvm.contrib import graph_runtime
from tvm.contrib.nvcc import have_fp16
from tvm.relay.testing.config import ctx_list

#filename = 'detection_model_vision_nms_800x450_cpu.pth'
#filename = 'det_COCO_e2e_mask_rcnn_DLA34_16_2e-2_90K_tvis_nms_1312x640_cpu.pth'
#filename = 'det_COCO_e2e_mask_rcnn_MobileNet_FPN_16_1e-2_90K_tvis_nms_1312x640_cpu.pth'
#filename = 'det_COCO_e2e_mask_rcnn_R_50_FPN_16_2e-2_90K_tvis_nms_1312x640_cpu.pth'
#filename = 'det_COCO_e2e_mask_rcnn_MobileNet_FPN_16_1e-2_90K_tvis_nms_1312x640_new_cpu.pth'
#filename = 'det_no_append_COCO_e2e_mask_rcnn_MobileNet_FPN_16_1e-2_90K_tvis_nms_1312x640_cpu.pth'


#filename = 'det_no_append_index_select_COCO_e2e_mask_rcnn_MobileNet_FPN_16_1e-2_90K_tvis_nms_1312x640_cpu.pth'
#filename = 'det_no_nonzero_COCO_e2e_mask_rcnn_MobileNet_FPN_16_1e-2_90K_tvis_nms_1312x640_cpu.pth'
#filename = 'det_pooler_no_put_nonzero_back_COCO_e2e_mask_rcnn_MobileNet_FPN_16_1e-2_90K_tvis_nms_1312x640_cpu.pth'
#filename = 'tempmodel_cpu.pth'

#These two should be the same, last from trevmor
#filename = 'model2_cpu.pth'
#filename = 'model10_cpu.pth'

#New one, only one class or something
filename = 'model11_cpu.pth'
#filename = 'model12_cpu.pth'

#filename = 'detection_model_vision_nms_800x450_cpu.pth'

scripted_model = torch.jit.load(filename)

"""
print('ts graph')
#print(scripted_model)
print(scripted_model.graph)
print('ts graph end')

print('ts nodes')
for node in scripted_model.graph.nodes():
    print(node)
print('ts nodes end')
"""

#11204264478797508617_2534701978479294800_2.pt

#input_name = 'input0'
#ishapes = [(1, 3, 450, 800)]
#ishapes = [(3,640,1312)]
ishapes = [(3,480,960)]
#input_data = torch.randn(input_shape)
#shape_list = [(input_name, input_data.shape)]

input_names = ["i{}".format(idx) for idx, ish in enumerate(ishapes)]
input_shapes = list(zip(input_names, ishapes))

inputs = [torch.randn(shape, dtype=torch.float)
          for shape in ishapes]

for i in inputs:
    print(type(i))
    input('get type')

mod, params = relay.frontend.from_pytorch(scripted_model, input_shapes)

#Should save or something so I don't have to wait to parse each time...

#print(mod)

with tvm.transform.PassContext(opt_level=3, disabled_pass=None):
    vm_exec = relay.vm.compile(mod, target="llvm", params=params)
from tvm.runtime.vm import VirtualMachine
vm = VirtualMachine(vm_exec, tvm.cpu())
#vm.init(tvm.cpu())
inputs1 = {}
for e, i in zip(input_names, inputs):
    inputs1[e] = tvm.nd.array(i)
result = vm.invoke("main", **inputs1)


print(result)


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.asnumpy()]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f))
        return result
    elif isinstance(o, tvm.relay.backend.interpreter.ConstructorValue):
        if o.constructor.name_hint == 'Cons':
            tl = vmobj_to_list(o.fields[1])
            hd = vmobj_to_list(o.fields[0])
            hd.extend(tl)
            return hd
        elif o.constructor.name_hint == 'Nil':
            return []
        elif 'tensor_nil' in o.constructor.name_hint:
            return [0]
        elif 'tensor' in o.constructor.name_hint:
            return [o.fields[0].asnumpy()]
        else:
            raise RuntimeError("Unknown object type: %s" %
                               o.constructor.name_hint)
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))

with torch.no_grad():
    pt_result = scripted_model(*inputs)

if not isinstance(pt_result, torch.Tensor):
    tvm_res = result.asnumpy().item()
    assert pt_result == tvm_res
else:
    tvm.testing.assert_allclose(result.asnumpy(), pt_result.numpy(),
                                rtol=1e-5, atol=1e-5)