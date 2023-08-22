import torch
import models
from timm.models import create_model
from models.modules.mobileone import reparameterize_model
from PIL import Image
import numpy as np
# To Train from scratch/fine-tuning
model = create_model("fastvit_ma36")
# ... train ...

# Load unfused pre-trained checkpoint for fine-tuning
# or for downstream task training like detection/segmentation
checkpoint = torch.load('model_zoo/fastvit_ma36.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
# ... train ...
img_np = np.array(Image.open('punch-suv-home-mob.png').convert('RGB'))
img_pt = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
# For inference
model.eval()
model(img_pt)
model_inf = reparameterize_model(model)