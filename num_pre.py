import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
import numpy as np
import torch
import os
import SimpleITK as sitk
from model.bra_unet import BRAUnet
from test_single import test_single_volume
from pathlib import Path
execute_in_docker =True
def inference():
    input_dir = Path("./input/images/pelvic-2d-ultrasound/") if execute_in_docker else Path("./test/")
    test_save_path = Path("./output/images/symphysis-segmentation/") if execute_in_docker else Path("./output/")
    # todo Load the trained model
    if execute_in_docker:
        snapshot = "./output/Psfh/best_model.pth"

    else:
        snapshot = "/model_weights/best_model1.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    snapshot = torch.load(snapshot, map_location=device)
    model = BRAUnet(img_size=256, num_classes=3, n_win=8, embed_dim=96)
    model.load_state_dict(snapshot)
    for filename in os.listdir(input_dir):
        image_path = os.path.join(str(input_dir), filename)
        image = sitk.ReadImage(image_path)
        filen = filename.split(".")[0]
        image = sitk.GetArrayFromImage(image)
        image = np.transpose(image, (1, 2, 0))
        image = np.expand_dims(image, axis=0)
        pred = test_single_volume(image, model,  patch_size=[256,256], test_save_path=test_save_path)
        pred = np.uint8(pred)
        pred = sitk.GetImageFromArray(pred)
        save_path = os.path.join(str(test_save_path),filen+'.mha')
        if not os.path.exists('./output/images/symphysis-segmentation'):
            os.makedirs('./output/images/symphysis-segmentation')
        pre=sitk.WriteImage(pred, save_path)
        print("Successful")
if __name__ == "__main__":
    inference()
    print("Successful")


