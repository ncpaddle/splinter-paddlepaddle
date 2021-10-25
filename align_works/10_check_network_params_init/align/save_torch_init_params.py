import json

from finetuning.modeling import ModelWithQASSHead, ClassificationHead
from finetuning_pytorch.modeling import ModelWithQASSHead as PyTorchModelWithQASSHead
import numpy as np
import torch
import pickle
import paddle
import random
from reprod_log import ReprodLogger, ReprodDiffHelper

random.seed(42)
np.random.seed(42)
paddle.seed(42)



model_torch = PyTorchModelWithQASSHead.from_pretrained("../../splinter", initialize_new_qass=True)
model_torch.train()

# torch.save(model_torch.state_dict(), '../../../splinter_init/pytorch_model.bin')

params = model_torch.state_dict()
for p in params:
    print(p)
