import sys
import torch
# import utils
import os
import numpy as np
# from torchvision import transforms
from PIL import Image
# import h5py
# import scipy.io as spio
# import nibabel as nib
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
# from IPython.display import display
# import torchvision

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('/u/fdammeier/repositories/NeuroFlow/script/sdxl')
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder

# # tf32 data type is faster than standard float32
# torch.backends.cuda.matmul.allow_tf32 = True

device = 'cuda'
## Need to download 'open_clip_pytorch_model.bin' from mindeyev2
clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="/u/fdammeier/checkpoints/mindeyev2/open_clip_pytorch_model.bin",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device).eval()

clip_seq_dim = 256
clip_emb_dim = 1664

class CLIP_Image_Dataset(Dataset):
    def __init__(self, image_path):
        self.img_data = image_path   #图像path

    def __getitem__(self, idx):
        img = Image.open(self.img_data[idx])  # 一张图像对应1个fmri
        img = TF.to_tensor(img).float()
        return img

    def __len__(self):
        return len(self.img_data)


data_path="/u/fdammeier/data/NeuroFlow/nsd"
batch_size=1000

for subj in [2, 5, 7]:
    save_path = os.path.join("/u/fdammeier/data/NeuroFlow/nsd", 'subj0{}'.format(subj))
    
    train_image_path = os.path.join(data_path, 'subj0{}/train_img'.format(subj))
    train_image = np.array([os.path.join(train_image_path, f'{i}.png') for i in range(len(os.listdir(train_image_path)))])
    
    print(f'Train Image Sub{subj}: {train_image.shape}')
    train_dataset = CLIP_Image_Dataset(train_image)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
    
    test_image_path = os.path.join(data_path, 'subj0{}/test_img'.format(subj))
        
    test_image = np.array([os.path.join(test_image_path, f'{i}.png') for i in range(len(os.listdir(test_image_path)))])
    print(f'Test Image Sub{subj}: {test_image.shape}')
    test_dataset = CLIP_Image_Dataset(test_image)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)
    mode = 'test'
    for test_i, image in enumerate(test_dataloader):
        print(test_i)
        
        with torch.no_grad():
            z = clip_img_embedder(image.to(device))
        
        if test_i == 0:
            zs = z.detach().cpu()
        else:
            zs = torch.cat((zs, z.detach().cpu()), dim=0)

    np.save(os.path.join(save_path, f'nsd_sdxl_clip_{mode}_sub{subj}.npy'), zs.numpy())
    del z, zs, image
    
    mode = 'train'
    for train_i, image in enumerate(train_dataloader):
        print(train_i)
        
        with torch.no_grad():
            z = clip_img_embedder(image.to(device))
        
        if train_i == 0:
            zs = z.detach().cpu()
        else:
            zs = torch.cat((zs, z.detach().cpu()), dim=0)

    np.save(os.path.join(save_path, f'nsd_sdxl_clip_{mode}_sub{subj}.npy'), zs.numpy())
    del z, zs, image