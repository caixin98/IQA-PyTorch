import argparse
import glob
import os
import csv
import torch
from skimage.metrics import structural_similarity as ssim

from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
import lpips
import cv2
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
def load_image(path):
    """ Load an image from a file path. """
    with Image.open(path) as img:
        return ToTensor()(img.convert("RGB"))
# def load_image(path):
#     img =  cv2.imread(path).astype("float")
#     img /= 255.
#     return img
def main():
    """Inference demo using PIQA metrics: PSNR, SSIM, and LPIPS."""
    parser = argparse.ArgumentParser(description="Compute IQA metrics using PIQA library.")
    parser.add_argument('-i', '--input', required=True, type=str, help='input image/folder path.')
    parser.add_argument('-r', '--ref', required=True, type=str, help='reference image path.')
    parser.add_argument('-m', '--metric_name', choices=['PSNR', 'SSIM', 'LPIPS'], default='PSNR', help='IQA metric name, case sensitive.')
    parser.add_argument('--save_file', type=str, help='path to save results.')

    args = parser.parse_args()

    # Initialize metrics
   
    metric_name = args.metric_name 
    metric = None
    if metric_name == 'PSNR':
        metric = psnr
    elif metric_name == 'SSIM':
        metric = ssim
    elif metric_name == 'LPIPS':
        metric = lpips

    if os.path.isfile(args.input):
            input_paths = [args.input]
            if args.ref is not None:
                ref_paths = [args.ref]
    else:
        input_paths = sorted(glob.glob(os.path.join(args.input, '*')))
        if args.ref is not None:
            ref_paths = sorted(glob.glob(os.path.join(args.ref, '*')))
    results = []
    test_img_num = len(input_paths)
    pbar = tqdm(total=test_img_num, unit='image')
    avg_score = 0
    for idx, img_path in enumerate(input_paths):
        query = load_image(img_path)
        ref_img_path = ref_paths[idx]
        reference = load_image(ref_img_path)

        # score = ssim(
        #             query, reference, multichannel=True, data_range=1.0,channel_axis = -1
        #         )
        # print(query.max(),query.min())
        
        score = loss_fn_alex(query.unsqueeze(0), reference.unsqueeze(0)).item()
        print(img_path, score)
        results.append((img_path, score))
        pbar.update(1)
        avg_score += score
    pbar.close()
    avg_score /= test_img_num

    msg = f'Average {metric_name} score of {args.input} with {test_img_num} images is: {avg_score}'
    print(msg)

    if args.save_file:
        with open(args.save_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image', 'Score'])
            for result in results:
                writer.writerow(result)
    else:
        for path, score in results:
            print(f"{path}: {args.metric_name} = {score}")

if __name__ == "__main__":
    main()