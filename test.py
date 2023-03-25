import os
import time
import numpy
import numpy as np
import torchvision.transforms as transforms
import scipy.io as scio
import torchvision
from option import opt
from decovNet import *

k_shape = [45, 45]
y_shape = [345, 345]
# k_shape = [23, 23]
# y_shape = [503, 343]
# k_shape = [45, 45]
# y_shape = [481, 321]

count =1

def models(name):
    if name == 'DecovNetShare':
        return DecovNetShare(filter_number=opt.filter_num, max_iter = opt.layer_num,
                                   random_initial=opt.random_initial)
    if name == 'DecovNetIndiviual':
        return DecovNetIndiviual(filter_number=opt.filter_num, max_iter = opt.layer_num,
                                           random_initial=opt.random_initial)
    if name == 'DecovNetConvergence':
        return DecovNetConvergence(filter_number=opt.filter_num, max_iter = opt.layer_num,
                                               convergence_type=opt.convergence_type,
                                               random_initial=opt.random_initial)

start_time = time.time()

def imageWrite(model):
    ssims = []
    psnrs = []

    pred_folder = opt.result_folder
    if not os.path.exists(pred_folder):
        os.mkdir(pred_folder)
    test_folder = opt.test_folder
    model.eval()

    with torch.no_grad():
        for item in os.listdir(test_folder):
            data = scio.loadmat(os.path.join(test_folder, item))
            x_dataset = torch.tensor(data["x"]).unsqueeze(0).unsqueeze(0)
            k_dataset = torch.tensor(data["kernel"]).unsqueeze(0).unsqueeze(0)
            y_dataset = torch.tensor(data["y"]).unsqueeze(0).unsqueeze(0)


            y_dataset = y_dataset.to(opt.device)
            k_dataset = k_dataset.to(opt.device)

            g=None
            u=None
            pred, g, u = model(y_dataset, k_dataset, g, u)
            targets_numpy = (x_dataset.cpu().numpy() * 255).squeeze()
            pred_numpy = (pred.cpu().numpy() * 255).squeeze()

            pred_numpy = pred_numpy[22: -22, 22: -22]
            targets_numpy = targets_numpy[22: -22, 22: -22]



            torchvision.utils.save_image(y_dataset, pred_folder + "/" + item.split(".")[0] + '_input.png')
            torchvision.utils.save_image(pred, pred_folder + "/" + item.split(".")[0] + '_pred.png')
            torchvision.utils.save_image(x_dataset, pred_folder + "/" + item.split(".")[0] + '_gt.png')

            # print(skimage.measure.compare_ssim(pred_numpy, targets_numpy, data_range=255),
            #         skimage.measure.compare_psnr(pred_numpy, targets_numpy, data_range=255))
            # ssims.append(skimage.measure.compare_ssim(pred_numpy, targets_numpy, data_range=255))
            # psnrs.append(skimage.measure.compare_psnr(pred_numpy, targets_numpy, data_range=255))
            # break

        # print(numpy.mean(ssims), numpy.mean(psnrs))
    return


if __name__ == "__main__":
    model = models(opt.model)
    model = model.to(opt.device)
    print(f'resume from {opt.model_dir + opt.trained_model}')

    assert os.path.exists(opt.model_dir + opt.trained_model)
    ckp = torch.load(opt.model_dir + opt.trained_model)
    model.load_state_dict(ckp['model'])

    imageWrite(model)
