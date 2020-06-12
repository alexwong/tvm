import torch
from torchvision.ops import roi_align, nms
from tvm import relay

scripted_model = torch.jit.load('detection_model_vision_nms_800x450_cpu.pth')

print(scripted_model)
print(scripted_model.graph)

input_name = 'input0'
input_shape = [1, 3, 450, 800]
input_data = torch.randn(input_shape)
shape_list = [(input_name, input_data.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

#print(mod)