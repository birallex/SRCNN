from skimage.measure import compare_ssim as ssim
from mse import mse
from psnr import psnr


def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, multichannel =True))
    return scores