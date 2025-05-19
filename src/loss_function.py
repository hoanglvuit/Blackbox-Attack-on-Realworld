import torch
import math
from utils import to_pytorch

class UnTargeted_idealW:
    def __init__(self, model, true):
        self.model = model
        self.true = true
        self.to_pytorch = to_pytorch

    def __call__(self, img):
        img_ = to_pytorch(img_)
        img_ = img_[None, :]
        preds = self.model.predict(img_).flatten()
        y = int(torch.argmax(preds))
        preds = preds.tolist()

        success = True if y != self.true else False

        f_true = math.log(math.exp(preds[self.true]) + 1e-30)
        preds[self.true] = -math.inf

        f_other = math.log(math.exp(max(preds)) + 1e-30)
        return [success, float(f_true - f_other)]