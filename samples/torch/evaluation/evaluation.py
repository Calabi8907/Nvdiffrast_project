import os
import pathlib
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
import csv

def load_img(path):
    img = Image.open(path).convert('RGB')
    # img = img.resize((256, 256))  # 필요시 해제/수정 가능
    transform = T.ToTensor()
    return transform(img)

def calc_psnr(img1, img2):
    np1 = (img1.permute(1,2,0).numpy() * 255).astype(np.uint8)
    np2 = (img2.permute(1,2,0).numpy() * 255).astype(np.uint8)
    return compare_psnr(np1, np2, data_range=255)

def calc_ssim(img1, img2):
    np1 = (img1.permute(1,2,0).numpy() * 255).astype(np.uint8)
    np2 = (img2.permute(1,2,0).numpy() * 255).astype(np.uint8)
    # shape print로 실제 크기 확인 (디버깅에 도움)
    print("SSIM input shape:", np1.shape, np2.shape)
    min_hw = min(np1.shape[0], np1.shape[1])
    win_size = min(7, min_hw)
    if win_size % 2 == 0:
        win_size -= 1
    try:
        # 최신 skimage (channel_axis)
        return compare_ssim(np1, np2, data_range=255, win_size=win_size, channel_axis=-1)
    except TypeError:
        # 구버전 호환 (multichannel)
        return compare_ssim(np1, np2, data_range=255, win_size=win_size, multichannel=True)


def calc_lpips(img1, img2, loss_fn):
    preprocess = lambda x: (x*2-1).unsqueeze(0)
    img1_p = preprocess(img1)
    img2_p = preprocess(img2)
    with torch.no_grad():
        dist = loss_fn(img1_p, img2_p)
    return dist.item()

def evaluate_and_save(base_dir, object_names, output_csv):
    
    
    loss_fn = lpips.LPIPS(net='vgg')  # .cuda() 사용 가능

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'ObjectName',
            'PSNR_ref_origin', 'SSIM_ref_origin', 'LPIPS_ref_origin',
            'PSNR_ref_opt', 'SSIM_ref_opt', 'LPIPS_ref_opt'
        ])

        for object_name in object_names:
            # 경로 개별 생성
            base_img_dir = os.path.join(base_dir, "samples", "torch", "output", object_name)
            ref_img_path = os.path.join(base_img_dir, "color_ref_downsampled.png")
            opt_img_path = os.path.join(base_img_dir, "result_tex.png")
            origin_img_path = os.path.join(base_img_dir, "origin_tex.png")

            # 존재 확인
            missing = []
            if not os.path.exists(ref_img_path): missing.append("ref")
            if not os.path.exists(opt_img_path): missing.append("opt")
            if not os.path.exists(origin_img_path): missing.append("origin")

            if missing:
                print(f"[경로 없음: {object_name}] - {', '.join(missing)}")
                continue

            img_gt = load_img(ref_img_path)
            img_opt = load_img(opt_img_path)
            img_origin = load_img(origin_img_path)

            # ref vs origin
            psnr_origin = calc_psnr(img_gt, img_origin)
            ssim_origin = calc_ssim(img_gt, img_origin)
            lpips_origin = calc_lpips(img_gt, img_origin, loss_fn)

            # ref vs opt
            psnr_opt = calc_psnr(img_gt, img_opt)
            ssim_opt = calc_ssim(img_gt, img_opt)
            lpips_opt = calc_lpips(img_gt, img_opt, loss_fn)

            print(f"{object_name}:")
            print(f"  [ref vs origin] PSNR={psnr_origin:.3f}, SSIM={ssim_origin:.3f}, LPIPS={lpips_origin:.4f}")
            print(f"  [ref vs opt   ] PSNR={psnr_opt:.3f}, SSIM={ssim_opt:.3f}, LPIPS={lpips_opt:.4f}")

            writer.writerow([
                object_name,
                f"{psnr_origin:.3f}", f"{ssim_origin:.3f}", f"{lpips_origin:.4f}",
                f"{psnr_opt:.3f}", f"{ssim_opt:.3f}", f"{lpips_opt:.4f}"
            ])

if __name__ == '__main__':
    base_dir = pathlib.Path(__file__).absolute().parents[3]
    object_names = ['book', 'box', 'corn_silk_tea', 'dior_lowtop', 'febreze','lays','mug','potato_chips']
    output_csv = os.path.join(base_dir, 'evaluation_results.csv')

    evaluate_and_save(base_dir, object_names, output_csv)
