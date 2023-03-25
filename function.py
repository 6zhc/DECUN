import torch

eps = 1e-31


def thresh_l1(x, lam):
    # Soft-thresholding for anisotropic total variation
    return torch.sign(x).mul(torch.maximum(x.abs() - lam, torch.tensor(0)))


def thresh_l2(gx, gy, lam):
    # Soft-thresholding for isotropic total variation
    g = (gx.pow(2) + gy.pow(2)).sqrt()
    zx = torch.maximum(g - lam, torch.tensor(0)).mul(gx).div(g + eps)
    zy = torch.maximum(g - lam, torch.tensor(0)).mul(gy).div(g + eps)
    return zx, zy


def thresh_l2_array(gArray, lam):
    # Soft-thresholding for isotropic total variation
    g = torch.sum(torch.stack([gItem.pow(2) for gItem in gArray]), dim=0).sqrt()
    z = [torch.maximum(g - lam, torch.tensor(0)).mul(gitem).div(g + eps) for gitem in gArray]
    return z
