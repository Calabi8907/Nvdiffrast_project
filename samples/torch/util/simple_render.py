import os
import pathlib
import argparse
import numpy as np
import torch
from PIL import Image
import math
import util
import nvdiffrast.torch as dr
from pygltflib import GLTF2

# Quaternion 곱셈 (오른손 좌표계 기준)
def q_mul(p, q):
    s1, V1 = p[0], p[1:]
    s2, V2 = q[0], q[1:]
    s = s1*s2 - np.dot(V1, V2)
    V = s1*V2 + s2*V1 + np.cross(V1, V2)
    return np.asarray([s, V[0], V[1], V[2]], np.float32)

# Quaternion -> 회전행렬 (4x4)
def q_to_mtx(q):
    w, x, y, z = q
    r00 = 1 - 2*y*y - 2*z*z
    r01 = 2*x*y - 2*z*w
    r02 = 2*x*z + 2*y*w
    r10 = 2*x*y + 2*z*w
    r11 = 1 - 2*x*x - 2*z*z
    r12 = 2*y*z - 2*x*w
    r20 = 2*x*z - 2*y*w
    r21 = 2*y*z + 2*x*w
    r22 = 1 - 2*x*x - 2*y*y
    rot = torch.tensor([
        [r00, r01, r02, 0],
        [r10, r11, r12, 0],
        [r20, r21, r22, 0],
        [  0,   0,   0, 1]
    ], dtype=torch.float32).cuda()
    return rot

# GLB 메타데이터 기반 보정
def apply_axis_fix(pos, glb_path):
    from pygltflib import GLTF2

    gltf = GLTF2().load(glb_path)
    generator = gltf.asset.generator.lower() if gltf.asset and gltf.asset.generator else ""
    rotation_quat = None
    for node in gltf.nodes:
        if node.rotation:
            rotation_quat = node.rotation
            break

    if "trimesh" in generator:
        # Trimesh는 Z-up + -Y Forward로 export됨 → Z축만 반전!
        pos[:, 2] *= -1
    elif "blender" in generator:
        # Blender는 이미 쿼터니언이 삽입되어 변환 완료! → 별도 반전 없음
        # (아무 처리도 하지 않음)
        
        print("BLENDER detected for", glb_path)
        # # X축 -90도 회전: Z-up → Y-up 변환
        # rot_x_neg90 = np.array([
        #     [1, 0, 0],
        #     [0, 0, -1],
        #     [0, 1, 0]
        # ], dtype=np.float32)
        # pos[:] = pos @ rot_x_neg90.T
        # print("BLENDER detected for", glb_path)
        # # pos[:, 2] *= -1  # Z축 반전
        # pos[:, 1] *= -1  # 상하 반전만 적용
        x = pos[:, 0].copy()
        y = pos[:, 1].copy()
        z = pos[:, 2].copy()
        pos[:, 0] = x
        pos[:, 1] = z
        pos[:, 2] = y
        pass
    else:
        # 기타 툴: 추가적인 분기가 필요할 수 있음 (여기서는 pass)
        pass

    return pos

# MVP 변환
def transform_pos(mtx, pos):
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, mtx.t())[None, ...]

# 렌더링 함수
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
                  quat=[1, 0, 0, 0],
                  translate=[0, 0, -3.5],
                  fov_deg=49.13):

    base_dir = pathlib.Path(__file__).absolute().parents[3]
    datadir = os.path.join(base_dir, "samples", "data", "npy_data", object_name)
    output_dir = os.path.join(base_dir, "samples", "torch", "output", object_name)
    img_dir = os.path.join(base_dir, "samples", "reference_data","png_seg")
    os.makedirs(out_dir, exist_ok=True)

    # 데이터 로딩
    pos_idx = np.load(os.path.join(datadir, "pos_idx.npy"))
    pos     = np.load(os.path.join(datadir, "pos.npy"))
    pos     = apply_axis_fix(pos, os.path.join(base_dir, "samples", "reference_data","glb_data", f"{object_name}.glb"))
    uv_idx  = np.load(os.path.join(datadir, "uv_idx.npy"))
    uv      = np.load(os.path.join(datadir, "uv.npy"))
    tex_img = Image.open(os.path.join(datadir, "tex.png")).convert("RGB")
    tex     = np.asarray(tex_img).astype(np.float32) / 255.0

    # 레퍼런스 이미지
    ref_img_path = os.path.join(img_dir, f"{object_name}_seg.png")
    ref_img = Image.open(ref_img_path).convert("RGB").resize((ref_res, ref_res), Image.LANCZOS)
    ref_img_np = np.asarray(ref_img).astype(np.float32) / 255.0

    # Tensor 변환
    pos_idx = torch.tensor(pos_idx, dtype=torch.int32).cuda()
    pos     = torch.tensor(pos, dtype=torch.float32).cuda()
    uv_idx  = torch.tensor(uv_idx, dtype=torch.int32).cuda()
    uv      = torch.tensor(uv, dtype=torch.float32).cuda()
    tex     = torch.from_numpy(tex).cuda()

    # Projection
    fov_rad = math.radians(fov_deg)
    x = math.tan(fov_rad / 2)
    proj = torch.tensor(util.projection(x=x, n=1.0, f=200.0), dtype=torch.float32).cuda()

    # View matrix (translation + rotation)
    tmat = torch.eye(4, dtype=torch.float32).cuda()
    tmat[:3, 3] = torch.tensor(translate, dtype=torch.float32)
    quat = torch.tensor(quat)
    rot_mat = q_to_mtx(quat)

    # 최종 MVP 행렬
    mvp = proj @ tmat @ rot_mat

    # 렌더링
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
    color = render(glctx, mvp, pos, pos_idx, uv, uv_idx, tex, res, enable_mip, max_mip_level)
    color_np = color[0].cpu().numpy()

    # 이미지 저장
    Image.fromarray((color_np * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{object_name}_rendered.png"))
    Image.fromarray((ref_img_np * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{object_name}_ref.png"))
    print(f"Saved: {object_name}_rendered.png and reference image")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_name", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--quat", type=float, nargs=4, default=[1, 0, 0, 0], help="w x y z")
    parser.add_argument("--translate", type=float, nargs=3, default=[0, 0, -3.5])
    parser.add_argument("--fov_deg", type=float, default=49.13)
    args = parser.parse_args()

    simple_render(object_name=args.object_name,
                  out_dir=args.out_dir,
                  quat=args.quat,
                  translate=args.translate,
                  fov_deg=args.fov_deg)