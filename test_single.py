import numpy as np
import torch
from scipy.ndimage import zoom
import copy
from PIL import Image
def test_single_volume(image, net, patch_size=[256, 256], test_save_path=None):
    image = image.squeeze(0)
    image = image.transpose(2,0,1)
    _, x, y = image.shape
    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).float().cpu()
    net=net.to('cpu')
    net.eval()
    with torch.no_grad():
        p1= net(input)
        out = torch.argmax(torch.softmax(p1, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        # 缩放预测结果图像同原始图像大小
        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out
    if test_save_path is not None:
        a1 = copy.deepcopy(prediction)
        prediction = Image.fromarray(np.uint8(a1)).convert('L')
    return prediction