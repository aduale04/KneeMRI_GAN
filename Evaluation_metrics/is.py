import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from glob import glob
import cv2

from tqdm import tqdm

# https://github.com/eyalbetzalel/inception-score-pytorch/blob/master/inception_score.py
def inception_score(imgs, batch_size=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    batch_size -- batch size for feeding into Inception v3
    """
    N = len(imgs)

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    py = np.mean(preds, axis=0)
    scores = []
    for i in range(preds.shape[0]):
        pyx = preds[i, :]
        scores.append(entropy(pyx, py))
    return np.exp(np.mean(scores))

if __name__ == "__main__":
    imgs = torch.tensor([cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob("generated/**/*.png", recursive=True) + glob("generated/**/*.jpg", recursive=True)])
    imgs = torch.permute(imgs, (0,3,1,2)) / 128 - 1
    print(inception_score(imgs))