import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from skimage import data
from sklearn.model_selection import train_test_split


class AtomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_x_path="process_plots/O/"):

        self.img = self.__readImages(img_x_path, startPoint=0)

        transform_list = [transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __readImages(self, imagePath, startPoint=0):

        imageData = []

        imagesNames = []

        for root, dirs, files in os.walk(imagePath):
            imagesNames = files[0:10]
            break

        for index in range(startPoint, len(imagesNames)):
            imageData.append(data.imread(imagePath + imagesNames[index]) / 256)

        imageData = np.array(imageData, dtype=np.float32).reshape(-1, 256, 256, 1)

        return imageData

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img = self.img[index]
        return img


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    epoch = 30
    testDataset = DataLoader(AtomDataset(),batch_size=1)
    g_model_out_path = "checkpoint/Model_G_{}.pth".format(epoch)

    # G_model = torch.load(g_model_out_path).eval()
    # if torch.cuda.is_available():
    #     G_model = G_model.cuda()

    for index, data in enumerate(testDataset):
        x = transforms.ToTensor()(data[0])
        # y = G_model(x)
        y = transforms.ToPILImage()(x)
        plt.imshow(y,cmap="gray")
        plt.show()