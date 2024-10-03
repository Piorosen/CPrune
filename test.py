#%%
import torchvision
import torch
import torchvision.models as models
from nni.compression.pytorch import ModelSpeedup
import torch.onnx


#%%
model = models.resnet18(pretrained=True).to('cpu')
# %%
model.load_state_dict(torch.load('/work/frontend/tmp_model.pth'))
masks_file = '/work/frontend/tmp_mask.pth'

dummy_input = torch.randn([1, 3, 224, 224])

m_speedup = ModelSpeedup(model, dummy_input, masks_file, 'cpu')
m_speedup.speedup_model()
# %%
m_speedup
# %%
# %%
torch.onnx.export(model,               # 실행될 모델
                  dummy_input,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "super_resolution.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=12,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

# %%
