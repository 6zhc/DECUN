import torch
import scipy.io as scio
import os
import numpy as np
from option import model_name, opt
from PIL import Image
from torchvision import transforms



class matDatasetedgeTapered(torch.utils.data.Dataset):
    def __init__(self, data_folder, k_shape, transform=None):
        self.x = []
        self.k = []
        self.y = []
        for file in os.listdir(data_folder):
            # print(file)
            data = scio.loadmat(os.path.join(data_folder, file))

            x_dataset = torch.tensor(data["x"]).unsqueeze(0).unsqueeze(0)
            y_dataset = torch.tensor(data["yEdgesTapered"])
            k_dataset = torch.tensor(data["kernel"])
            #
            # k_dataset = torch.nn.functional.pad(k_dataset,
            #                                     (int((k_shape[0] - k_dataset.shape[0])/2),
            #                                      int((k_shape[0] - k_dataset.shape[0]+1) / 2),
            #                                      int((k_shape[1] - k_dataset.shape[1]) / 2),
            #                                      int((k_shape[1] - k_dataset.shape[1]+1) / 2)),
            #                                     mode='constant', value=0)
            k_dataset = k_dataset.unsqueeze(0).unsqueeze(0)

            # y_dataset = torch.conv2d(x_dataset, k_dataset,
            #                          padding=(k_dataset.shape[2] - 1, k_dataset.shape[3] - 1))

            # y_dataset = torch.conv2d(torch.nn.functional.pad(x_dataset,
            #     (int((k_dataset.shape[2]-1)/2), int((k_dataset.shape[2]-1)/2),
            #      int((k_dataset.shape[3]-1)/2), int((k_dataset.shape[3]-1)/2)), mode='circular')
            #                          , torch.tensor(data["f"]).unsqueeze(0).unsqueeze(0),
            #                          padding="valid")
            # y_shape = y_dataset.squeeze().unsqueeze(0).shape
            # y_dataset += 0.01 * np.random.rand(*y_shape)
            self.x.append(x_dataset.squeeze().unsqueeze(0))
            self.y.append(y_dataset.squeeze().unsqueeze(0))
            self.k.append(k_dataset.squeeze().unsqueeze(0))
        # self.x =torch.stack(self.x)
        # self.y =torch.stack(self.y)
        # self.k =torch.stack(self.k)

        self.length = len(os.listdir(data_folder))

        self.transform = transform

        # y_dataset = Image.open("blurred_image/im05_flit02.mat.bmp").convert('L')
        # x_dataset = Image.open("deblurred_image/im05_flit02.mat.bmp").convert('L')
        # k_dataset = Image.open("blur_kernal/im05_flit02.mat.bmp").convert('L')
        # test_dataset = Image.open("temp.png").convert('L')

        # image2tensor = transforms.ToTensor()
        # x_dataset = image2tensor(x_dataset).squeeze()
        # k_dataset = image2tensor(k_dataset).squeeze()
        # y_dataset = image2tensor(y_dataset).squeeze()
        # test_dataset = image2tensor(test_dataset).squeeze()

    def __getitem__(self, index):
        # return self.y[index], self.k[index], self.x[index]

        k_index = torch.randint(self.length, (1,))[0]
        x_dataset = self.x[index].to(opt.device)
        k_dataset = self.k[k_index].to(opt.device)
        y_dataset = torch.conv2d(x_dataset.unsqueeze(0), k_dataset.unsqueeze(0).rot90(2, (-2, -1)),
                                 padding=(0, 0))
        y_dataset += torch.normal(mean=0, std=opt.noise, size=y_dataset.shape).to(opt.device)
        y_dataset = y_dataset.squeeze().unsqueeze(0)
        return y_dataset, k_dataset, x_dataset

    def __len__(self):
        return self.length

class matDatasetedgeFilename(torch.utils.data.Dataset):
    def __init__(self, data_folder, k_shape, transform=None):
        self.data_folder = data_folder
        self.x = os.listdir(data_folder + '/sharp/')
        self.k = os.listdir(data_folder + '/kernel/')

        self.length = len(self.x)
        self.k_length = len(self.k)

        self.transform = transform

        # y_dataset = Image.open("blurred_image/im05_flit02.mat.bmp").convert('L')
        # x_dataset = Image.open("deblurred_image/im05_flit02.mat.bmp").convert('L')
        # k_dataset = Image.open("blur_kernal/im05_flit02.mat.bmp").convert('L')
        # test_dataset = Image.open("temp.png").convert('L')

        # image2tensor = transforms.ToTensor()
        # x_dataset = image2tensor(x_dataset).squeeze()
        # k_dataset = image2tensor(k_dataset).squeeze()
        # y_dataset = image2tensor(y_dataset).squeeze()
        # test_dataset = image2tensor(test_dataset).squeeze()

    def __getitem__(self, index):
        # return self.y[index], self.k[index], self.x[index]

        image2tensor = transforms.ToTensor()

        k_index = torch.randint(self.k_length, (1,))[0]
        x_dataset = Image.open(self.data_folder + '/sharp/' + self.x[index]).convert('L')
        x_dataset = np.array(x_dataset) /255
        x_dataset = image2tensor(x_dataset).to(opt.device)
        k_dataset = Image.open(self.data_folder + '/kernel/' + self.k[k_index]).convert('L')
        k_dataset = np.array(k_dataset)/np.sum(np.sum(k_dataset))
        k_dataset = image2tensor(k_dataset).to(opt.device)

        y_dataset = torch.conv2d(x_dataset.unsqueeze(0), k_dataset.unsqueeze(0).rot90(2, (-2, -1)),
                                 padding=(0, 0))

        # print(y_dataset)
        # print(self.x[index])
        y_dataset += torch.normal(mean=0, std=opt.noise, size=y_dataset.shape).to(opt.device)
        y_dataset = y_dataset.squeeze().unsqueeze(0)
        return y_dataset, k_dataset, x_dataset

    def __len__(self):
        return self.length