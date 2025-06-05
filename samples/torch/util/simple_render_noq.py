import os
import pathlib
import argparse
import numpy as np
import torch
from PIL import Image
import math
import util
import nvdiffrast.torch as dr

# Transform vertex positions by MVP matrix
def transform_pos(mtx, pos):
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, mtx.t())[None, ...]

# Rasterize and render
def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')
    color = color * torch.clamp(rast_out[..., -1:], 0, 1)
    return color

def simple_render(object_name,
                  out_dir='output',
                  ref_res=512,
                  res=512,
                  enable_mip=True,
                  max_mip_level=9,
                  use_opengl=False,
                  translate=[0, 0, -3.5],
                  fov_deg=49.13):

    base_dir = pathlib.Path(__file__).absolute().parents[0]
    datadir = os.path.join(base_dir, "data", "npy_data", object_name)
    img_dir = os.path.join(base_dir, "samples", "img_data")
    os.makedirs(out_dir, exist_ok=True)

    # Load model npy/texture
    pos_idx = np.load(os.path.join(datadir, "pos_idx.npy"))
    pos     = np.load(os.path.join(datadir, "pos.npy"))
    uv_idx  = np.load(os.path.join(datadir, "uv_idx.npy"))
    uv      = np.load(os.path.join(datadir, "uv.npy"))
    tex_img = Image.open(os.path.join(datadir, "tex.png")).convert("RGB")
    tex     = np.asarray(tex_img).astype(np.float32) / 255.0

    # Load reference image
    ref_img_path = os.path.join(img_dir, f"{object_name}_seg.png")
    ref_img = Image.open(ref_img_path).convert("RGB").resize((ref_res, ref_res), Image.LANCZOS)
    ref_img_np = np.asarray(ref_img).astype(np.float32) / 255.0
    ref_img_tensor = torch.from_numpy(ref_img_np).cuda().unsqueeze(0)

    # Tensorize inputs
    pos_idx = torch.tensor(pos_idx, dtype=torch.int32).cuda()
    pos     = torch.tensor(pos, dtype=torch.float32).cuda()
    uv_idx  = torch.tensor(uv_idx, dtype=torch.int32).cuda()
    uv      = torch.tensor(uv, dtype=torch.float32).cuda()
    tex     = torch.from_numpy(tex).cuda()

    # Projection matrix
    fov_rad = math.radians(fov_deg)
    x = math.tan(fov_rad / 2)
    proj = torch.tensor(util.projection(x=x, n=1.0, f=200.0), dtype=torch.float32).cuda()

    # View matrix (translation only)
    tmat = torch.eye(4, dtype=torch.float32).cuda()
    tmat[:3, 3] = torch.tensor(translate, dtype=torch.float32)

    # MVP
    mvp = proj @ tmat

    # Render
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
    color = render(glctx, mvp, pos, pos_idx, uv, uv_idx, tex, res, enable_mip, max_mip_level)
    color_np = color[0].cpu().numpy()

    # Save output
    Image.fromarray((color_np * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{object_name}_rendered.png"))
    Image.fromarray((ref_img_np * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{object_name}_ref.png"))
    print(f"Saved: {object_name}_rendered.png and reference image")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_name", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--translate", type=float, nargs=3, default=[0, 0, -3.5])
    parser.add_argument("--fov_deg", type=float, default=49.13)
    args = parser.parse_args()

    simple_render(object_name=args.object_name,
                  out_dir=args.out_dir,
                  translate=args.translate,
                  fov_deg=args.fov_deg)
