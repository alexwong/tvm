import torch
from torchvision.ops import roi_align, nms
from tvm import relay

#filename = 'detection_model_vision_nms_800x450_cpu.pth'
#filename = 'det_COCO_e2e_mask_rcnn_DLA34_16_2e-2_90K_tvis_nms_1312x640_cpu.pth'
filename = 'det_COCO_e2e_mask_rcnn_MobileNet_FPN_16_1e-2_90K_tvis_nms_1312x640_cpu.pth'
#filename = 'det_COCO_e2e_mask_rcnn_R_50_FPN_16_2e-2_90K_tvis_nms_1312x640_cpu.pth'

scripted_model = torch.jit.load(filename)

input_name = 'input0'
input_shape = [1, 3, 450, 800]
#input_shape = [1,3,640,1312]
input_data = torch.randn(input_shape)
shape_list = [(input_name, input_data.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)