from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import config
import torchvision
import matplotlib.pyplot as plt

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations['image'], augmentations['image0']
        input_image = config.transform_only_input(image=input_image)['image']
        target_image = config.transform_only_mask(image=target_image)['image']

        return input_image, target_image

if __name__ == '__main__':
    dataset = MapDataset('maps/train')
    dataloader = DataLoader(dataset, 1)
    for i, (input_image, target_image) in enumerate(dataloader):
        input_image = input_image.squeeze(0)
        img = input_image.numpy() # FloatTensor转为ndarray
        img = np.transpose(img, (1,2,0)) # 把channel那一维放到最后
        target_image = target_image.squeeze(0)
        target_image = target_image.numpy() # FloatTensor转为ndarray
        target_image = np.transpose(target_image, (1,2,0)) # 把channel那一维放到最后
        # 显示图片
        plt.imshow(img)
        plt.show()
        plt.imshow(target_image)
        plt.show()
        break
