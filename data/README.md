# BRAU-Net
The codes for the work "Pubic Symphysis-Fetal Head Segmentation Using Pure Transformer with Bi-level Routing Attention"(https://arxiv.org/pdf/2310.00289.pdf). I hope this will help you to reproduce the results.

## 1. Prepare data
- We convert each training image and mask into a npz file. Get processed train data in this link. (https://drive.google.com/file/d/1HPy_4OrMWCn0g3JYIJlpwBKXsiEBxQAY/view?usp=sharing)

## 2. Environment
- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
  
## 3. Train/Test

- Run the train script on PS-FH-AOP dataset. The batch size we used is 4. 

- Train

```bash
python train.py --dataset Psfh  --root_path your DATA_DIR --max_epochs 100 --output_dir your OUT_DIR  --img_size 256 --base_lr 0.001 --batch_size 4
```

- Test 

```bash
python num_pre.py
```

## References
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [BiFormer](https://github.com/rayleizhu/BiFormer)

