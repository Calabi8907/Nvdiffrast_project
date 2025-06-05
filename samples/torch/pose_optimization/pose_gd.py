# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import pathlib
import sys
import numpy as np
import torch
from PIL import Image
import imageio
import math


UTIL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'util')
sys.path.append(UTIL_DIR)
from util import util

import nvdiffrast.torch as dr
from pygltflib import GLTF2

# from Perceptual_loss import PerceptualLoss
# from combined_loss_module import combined_loss, gradient_loss



#사용법
# ~.py  --object_name [객체명 (ex.box)]







#----------------------------------------------------------------------------
# Quaternion math.
#----------------------------------------------------------------------------


# Unit quaternion.
def q_unit():
    return np.asarray([1, 0, 0, 0], np.float32)

# Get a random normalized quaternion.
def q_rnd():
    u, v, w = np.random.uniform(0.0, 1.0, size=[3])
    v *= 2.0 * np.pi
    w *= 2.0 * np.pi
    return np.asarray([(1.0-u)**0.5 * np.sin(v), (1.0-u)**0.5 * np.cos(v), u**0.5 * np.sin(w), u**0.5 * np.cos(w)], np.float32)

# Get a random quaternion from the octahedral symmetric group S_4.
_r2 = 0.5**0.5
_q_S4 = [[ 1.0, 0.0, 0.0, 0.0], [ 0.0, 1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0, 1.0],
         [-0.5, 0.5, 0.5, 0.5], [-0.5,-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5, 0.5], [ 0.5, 0.5,-0.5, 0.5],
         [ 0.5, 0.5, 0.5, 0.5], [-0.5, 0.5,-0.5, 0.5], [ 0.5,-0.5,-0.5, 0.5], [-0.5,-0.5, 0.5, 0.5],
         [ _r2,-_r2, 0.0, 0.0], [ _r2, _r2, 0.0, 0.0], [ 0.0, 0.0, _r2, _r2], [ 0.0, 0.0,-_r2, _r2],
         [ 0.0, _r2, _r2, 0.0], [ _r2, 0.0, 0.0,-_r2], [ _r2, 0.0, 0.0, _r2], [ 0.0,-_r2, _r2, 0.0],
         [ _r2, 0.0, _r2, 0.0], [ 0.0, _r2, 0.0, _r2], [ _r2, 0.0,-_r2, 0.0], [ 0.0,-_r2, 0.0, _r2]]
def q_rnd_S4():
    return np.asarray(_q_S4[np.random.randint(24)], np.float32)

# Quaternion slerp.
def q_slerp(p, q, t):
    d = np.dot(p, q)
    if d < 0.0:
        q = -q
        d = -d
    if d > 0.999:
        a = p + t * (q-p)
        return a / np.linalg.norm(a)
    t0 = np.arccos(d)
    tt = t0 * t
    st = np.sin(tt)
    st0 = np.sin(t0)
    s1 = st / st0
    s0 = np.cos(tt) - d*s1
    return s0*p + s1*q

# Quaterion scale (slerp vs. identity quaternion).
def q_scale(q, scl):
    return q_slerp(q_unit(), q, scl)

# Quaternion product.
def q_mul(p, q):
    s1, V1 = p[0], p[1:]
    s2, V2 = q[0], q[1:]
    s = s1*s2 - np.dot(V1, V2)
    V = s1*V2 + s2*V1 + np.cross(V1, V2)
    return np.asarray([s, V[0], V[1], V[2]], np.float32)

# Angular difference between two quaternions in degrees.
def q_angle_deg(p, q):
    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()
    d = np.abs(np.dot(p, q))
    d = min(d, 1.0)
    return np.degrees(2.0 * np.arccos(d))

# Quaternion product
def q_mul_torch(p, q):
    a = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    b = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    c = p[0]*q[2] + p[2]*q[0] + p[3]*q[1] - p[1]*q[3]
    d = p[0]*q[3] + p[3]*q[0] + p[1]*q[2] - p[2]*q[1]
    return torch.stack([a, b, c, d])

# Convert quaternion to 4x4 rotation matrix.
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
    
    # 모든 연산을 torch 텐서로 유지
    rot = torch.stack([
        torch.stack([r00, r01, r02, torch.tensor(0.0, device=q.device)]),
        torch.stack([r10, r11, r12, torch.tensor(0.0, device=q.device)]),
        torch.stack([r20, r21, r22, torch.tensor(0.0, device=q.device)]),
        torch.stack([torch.tensor(0.0, device=q.device), 
                    torch.tensor(0.0, device=q.device), 
                    torch.tensor(0.0, device=q.device), 
                    torch.tensor(1.0, device=q.device)])
    ])
    return rot

# 각도 -> 쿼터니안 변환
def euler_xyz_to_quaternion(rx, ry, rz):
    cx, cy, cz = np.cos([rx/2, ry/2, rz/2])
    sx, sy, sz = np.sin([rx/2, ry/2, rz/2])
    qw = cx*cy*cz + sx*sy*sz
    qx = sx*cy*cz - cx*sy*sz
    qy = cx*sy*cz + sx*cy*sz
    qz = cx*cy*sz - sx*sy*cz
    return np.array([qw, qx, qy, qz], dtype=np.float32)

#----------------------------------------------------------------------------
# Helpers.
# 카메라 해상도 및 개별 픽셀 크기 정보로 센서 width, height 정보 추론하는 코드 필요 (추후 카메라 정보로부터 정보를 받는다면)

# 카메라에 따른 fov 추정 로직
def compute_fov_y(focal_length_equiv_mm, sensor_width_mm, sensor_height_mm):
    """
    센서 크기와 등가 초점 거리로부터 실제 초점 거리 및 수직 FOV 계산

    Args:
        sensor_width_mm (float): 센서 가로 길이 (mm)
        sensor_height_mm (float): 센서 세로 길이 (mm)
        focal_length_equiv_mm (float): 35mm 등가 초점 거리 (mm)

    Returns:
        dict: {
            'sensor_diagonal': 센서 대각선 길이 (mm),
            'crop_factor': 크롭 팩터,
            'focal_length_actual': 실제 초점 거리 (mm),
            'fov_y_deg': 수직 시야각 (degree)
        }
    """
    # 1. 센서 대각선
    sensor_diagonal = math.sqrt(sensor_width_mm**2 + sensor_height_mm**2)

    # 2. 크롭 팩터 (기준 대각선 = 43.3mm)
    crop_factor = 43.3 / sensor_diagonal

    # 3. 실제 초점 거리
    focal_length_actual = focal_length_equiv_mm / crop_factor

    # 4. 수직 FOV 계산
    fov_y_rad = 2 * math.atan(sensor_height_mm / (2 * focal_length_actual))
    fov_y_deg = math.degrees(fov_y_rad)

    return fov_y_deg

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    return color

def make_grid(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

# glb 파일에 따른 축 보정 처리
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
        x = pos[:, 0].copy()
        y = pos[:, 1].copy()
        z = pos[:, 2].copy()
        pos[:, 0] = x
        pos[:, 1] = -y
        pos[:, 2] = z
        # pass
    elif "blender" in generator:
        x = pos[:, 0].copy()
        y = pos[:, 1].copy()
        z = pos[:, 2].copy()
        pos[:, 0] = x
        pos[:, 1] = z
        pos[:, 2] = y
        # pass
    else:
        # 기타 툴: 추가적인 분기가 필요할 수 있음 (여기서는 pass)
        pass
    return pos

#----------------------------------------------------------------------------

def silhouette_loss(img1, img2, threshold=0.1):
    """
    실루엣 로스 계산 함수
    Args:
        img1, img2: (B, H, W, 3) 형태의 이미지 텐서
        threshold: 배경과 객체를 구분하는 임계값
    Returns:
        실루엣 로스 값
    """
    # 이미지를 그레이스케일로 변환
    gray1 = 0.299 * img1[..., 0] + 0.587 * img1[..., 1] + 0.114 * img1[..., 2]
    gray2 = 0.299 * img2[..., 0] + 0.587 * img2[..., 1] + 0.114 * img2[..., 2]
    
    # 마스크 생성 (threshold보다 큰 값은 1, 작은 값은 0)
    mask1 = (gray1 > threshold).float()
    mask2 = (gray2 > threshold).float()
    
    # 실루엣 로스 계산 (마스크 간의 차이)
    loss = torch.mean((mask1 - mask2) ** 2)
    return loss

def fit_pose(object_name,
            y_deg_base,
            pitch_base,
            roll_base,
            translate_z_base,
            translate_y_base,
            translate_x_base,
            max_iter=20000,
            log_interval=10,
            display_interval=None,
            display_res=512,
            enable_mip=True,
            res=512,
            # res=2048,
            ref_res=2048,
            out_dir=None,
            log_fn=None,
            use_opengl=False):

    log_file = None
    base_dir = pathlib.Path(__file__).absolute().parents[3]
    
    # out_dir 관련 파라미터 전부 주석처리.
    # if out_dir:
    #     os.makedirs(out_dir, exist_ok=True)
    #     if log_fn:
    #         log_file = open(os.path.join(out_dir, log_fn), 'wt')
    # else:
    #     imgsave_interval, texsave_interval = None, None

# 데이터 디렉토리 설정
    datadir = os.path.join(base_dir, "samples", "data", "npy_data", object_name)
    output_dir = os.path.join(base_dir, "samples", "torch", "output", object_name, "output_pose_gd")
    os.makedirs(output_dir, exist_ok=True)

# obj 파일 load 부분
    # 데이터 로딩
    pos_idx = np.load(os.path.join(datadir, "pos_idx.npy"))
    pos     = np.load(os.path.join(datadir, "pos.npy"))
    #파일 성격에 맞게 축 보정. (파일이 어떤 프로그램을 통해서 npy로 변환되었는지는 util\inspect_glb_metadata 활용용)
    pos     = apply_axis_fix(pos, os.path.join(base_dir, "samples", "reference_data","glb_data", f"{object_name}.glb"))
    uv_idx  = np.load(os.path.join(datadir, "uv_idx.npy"))
    uv      = np.load(os.path.join(datadir, "uv.npy"))

    tex_img = Image.open(os.path.join(datadir, "tex.png")).convert("RGB")
    tex     = np.asarray(tex_img).astype(np.float32) / 255.0
    tex_o   = tex.copy()
    max_mip_level = 9

    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))

    # Ref 이미지 로드
    img_dir = os.path.join(base_dir, "samples", "reference_data","png_seg")
    ref_img_path = os.path.join(img_dir, f"{object_name}_seg.png")
    ref_img = Image.open(ref_img_path).convert("RGB").resize((ref_res, ref_res), Image.LANCZOS)
    ref_img_np = np.asarray(ref_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(ref_img_np).cuda().unsqueeze(0)

    #output_dir에 어떤 ref와 loss가 계산되는지 직관적으로 보기 위함.
    Image.fromarray((ref_img_np * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{object_name}_ref.png"))


# 로드한 데이터 텐서 생성
    # Some input geometry contains vertex positions in (N, 4) (with v[:,3]==1).  Drop
    # the last column in that case.
    if pos.shape[1] == 4: pos = pos[:, 0:3]

    # Create position/triangle index tensors
    pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda()
    vtx_pos = torch.from_numpy(pos.astype(np.float32)).cuda()
    uv_idx  = torch.from_numpy(uv_idx.astype(np.int32)).cuda()
    vtx_uv  = torch.from_numpy(uv.astype(np.float32)).cuda()

    tex     = torch.from_numpy(tex.astype(np.float32)).cuda()
    # tex_opt = torch.full(tex.shape, 0.2, device='cuda', requires_grad=True)
    tex_opt = torch.from_numpy(tex_o.astype(np.float32)).cuda().requires_grad_()
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    # Projection matrix
    proj = util.projection(x=0.4, n=1.0, f=200.0)







    # 초기 카메라 값 하드코딩.
    # Initialize pose parameters

# 1. book의 경우, 아래 범위에서 랜덤 샘플링
    # book 탐색 범위
    # yaw	: -10 ~ 10 (y_deg)
    # pitch 	: 300 ~ 330
    # roll 	: -3 ~ 10
    # z	: 1.5 ~ 1.9
    # y 	: 0.03 ~ 0.07
    # x	: -0.05 ~ 0.06
    # Base values for each parameter

    # y_deg_base = -5
    # pitch_base = 315
    # roll_base = 2.0
    # translate_z_base = 1.8
    # translate_y_base = 0.04
    # translate_x_base = 0.0

    # Range values for each parameter
    y_deg_range = 5.0
    pitch_range = 10.0
    roll_range = 2.0
    translate_z_range = 0.1
    translate_y_range = 0.01
    translate_x_range = 0.01

    # color_opt_loss_0.022404_ydeg_-4.32_pitch_313.02_roll_2.73_z_1.786721_y_0.040217_x_0.030147
    # y_deg, pitch, roll, translate_z, translate_y, translate_x = -4.32, 313.02, 2.73, 1.786721, 0.040217, 0.030147 # base
    # y_deg, pitch, roll, translate_z, translate_y, translate_x = -9.32, 303.02, 0.73, 1.686721, 0.030217, 0.020147 # min
    # y_deg, pitch, roll, translate_z, translate_y, translate_x = 0.68, 323.02, 4.73, 1.886721, 0.050217, 0.040147 # max
    
    # y_deg, pitch, roll, translate_z, translate_y, translate_x = -5.0, 315.0, 2.0, 1.8, 0.04, 0.0 # base
    # y_deg, pitch, roll, translate_z, translate_y, translate_x = -10.0, 300.0, -3.0, 1.5, 0.03, -0.05 # min
    # y_deg, pitch, roll, translate_z, translate_y, translate_x = 0.0, 330.0, 7.0, 1.9, 0.07, 0.05 # max

# 2. corn_silk_tea
    # y_deg_base = 120
    # pitch_base = 30
    # roll_base = 26
    # translate_z_base = 1.5
    # translate_y_base = 0.0
    # translate_x_base = 0.0

    

# 3. box
    # y_deg_base = -5
    # pitch_base = 310
    # roll_base = 3
    # translate_z_base = 1.7
    # translate_y_base = 0.0
    # translate_x_base = 0.02

    
# 3. dior_lowtop
    # y_deg_base = -2
    # pitch_base = -20
    # roll_base = -1
    # translate_z_base = 1.6
    # translate_y_base = -0.01
    # translate_x_base = 0.03

# 3. tumbler
#     y_deg_base = 18
#     pitch_base = -25
#     roll_base = -8
#     translate_z_base = 0.4
#     translate_y_base = 0.04
#     translate_x_base = 0.03

# 4. lays
#     y_deg_base = -2
#     pitch_base = -2
#     roll_base = 0
#     translate_z_base = 2.0
#     translate_y_base = -0.01
#     translate_x_base = 0.00

# 5. mug
#     y_deg_base = 0
#     pitch_base = -13
#     roll_base = 0
#     translate_z_base = 1.75
#     translate_y_base = 0.00
#     translate_x_base = 0.00

# # 6. potato_chips
#     y_deg_base = 0
#     pitch_base = 0
#     roll_base = 0
#     translate_z_base = 1.75
#     translate_y_base = 0.00
#     translate_x_base = 0.00

# # 6. febreze
#     y_deg_base = -6
#     pitch_base = -10
#     roll_base = 0
#     translate_z_base = 1.75
#     translate_y_base = 0.00
#     translate_x_base = 0.00


    # # Generate random values within ranges
    # y_deg = np.random.uniform(y_deg_base - y_deg_range, y_deg_base + y_deg_range)
    # pitch = np.random.uniform(pitch_base - pitch_range, pitch_base + pitch_range)
    # roll = np.random.uniform(roll_base - roll_range, roll_base + roll_range)
    # translate_z = np.random.uniform(translate_z_base - translate_z_range, translate_z_base + translate_z_range)
    # translate_y = np.random.uniform(translate_y_base - translate_y_range, translate_y_base + translate_y_range)
    # translate_x = np.random.uniform(translate_x_base - translate_x_range, translate_x_base + translate_x_range)
    



    # Generate random values within ranges
    y_deg = y_deg_base
    pitch = pitch_base
    roll = roll_base
    translate_z = translate_z_base
    translate_y = translate_y_base
    translate_x = translate_x_base








    # Convert to torch tensors with explicit float32 dtype
    y_deg = torch.tensor(y_deg, dtype=torch.float32, device='cuda', requires_grad=True)
    pitch = torch.tensor(pitch, dtype=torch.float32, device='cuda', requires_grad=True)
    roll = torch.tensor(roll, dtype=torch.float32, device='cuda', requires_grad=True)
    translate_z = torch.tensor(translate_z, dtype=torch.float32, device='cuda', requires_grad=True)
    translate_y = torch.tensor(translate_y, dtype=torch.float32, device='cuda', requires_grad=True)
    translate_x = torch.tensor(translate_x, dtype=torch.float32, device='cuda', requires_grad=True)

    # Initialize optimizer with different learning rates for rotation and translation
    optimizer = torch.optim.Adam([
        {'params': [y_deg, pitch, roll], 'lr': 1e-1},  # 회전 파라미터: 더 큰 learning rate
        {'params': [translate_z, translate_y, translate_x], 'lr': 1e-3}  # 이동 파라미터: 더 작은 learning rate
    ], betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)

    best_loss = float('inf')
    best_params = None

    for it in range(max_iter):
        optimizer.zero_grad()

        # Convert degrees to radians and maintain gradients
        ry = y_deg * (math.pi / 180.0)
        rx = pitch * (math.pi / 180.0)
        rz = roll * (math.pi / 180.0)

        # Create quaternion from Euler angles (maintaining gradients)
        half_angles = torch.stack([rx/2, ry/2, rz/2])
        cx, cy, cz = torch.cos(half_angles)
        sx, sy, sz = torch.sin(half_angles)
        
        # 쿼터니언 계산 (그래디언트 유지)
        qw = cx*cy*cz + sx*sy*sz
        qx = sx*cy*cz - cx*sy*sz
        qy = cx*sy*cz + sx*cy*sz
        qz = cx*cy*sz - sx*sy*cz
        
        # 정규화 (그래디언트 유지)
        q = torch.stack([qw, qx, qy, qz])
        q_norm = torch.norm(q)
        q = q / q_norm

        # Create transformation matrix
        translate_matrix = torch.eye(4, dtype=torch.float32, device='cuda')
        translate_matrix[2, 3] = -3.5 + translate_z
        translate_matrix[1, 3] = translate_y
        translate_matrix[0, 3] = translate_x
        mvp = torch.matmul(torch.tensor(proj, dtype=torch.float32, device='cuda'), translate_matrix)
        mtx_total = torch.matmul(mvp, q_to_mtx(q))

        # Render
        color_opt = render(glctx, mtx_total, vtx_pos, pos_idx, vtx_uv, uv_idx,
                         tex_opt, res, enable_mip, max_mip_level)
        color_ref = img_tensor
        while color_ref.shape[1] > res:
            color_ref = util.bilinear_downsample(color_ref)

        # Calculate loss
        diff = (color_opt - color_ref) ** 2
        diff = torch.tanh(5.0 * torch.max(diff, dim=-1)[0])
        l2_loss = torch.mean(diff)
        threshold = 0.01
        sil_loss = silhouette_loss(color_opt, color_ref, threshold)
        alpha = 0.7
        loss = (1.0 - alpha) * l2_loss + alpha * sil_loss

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Update learning rate
        # scheduler.step(loss)

        # Save best parameters
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {
                'y_deg': y_deg.item(),
                'pitch': pitch.item(),
                'roll': roll.item(),
                'translate_z': translate_z.item(),
                'translate_y': translate_y.item(),
                'translate_x': translate_x.item()
            }

            # Save best image
            with torch.no_grad():
                color_opt_np = color_opt[0].cpu().numpy()
                # color_opt_img = (color_opt_np * 255.0).clip(0, 255).astype(np.uint8)
                # Image.fromarray(color_opt_img).save(os.path.join(output_dir, 
                #     f'color_opt_loss_{best_loss:.6f}_ydeg_{y_deg.item():.2f}_pitch_{pitch.item():.2f}_roll_{roll.item():.2f}_z_{translate_z.item():.6f}_y_{translate_y.item():.6f}_x_{translate_x.item():.6f}.png'))

        # Log progress
        if it % log_interval == 0:
            print(f"Iteration {it}, Loss: {loss.item():.6f}, Best Loss: {best_loss:.6f}")
            print(f"Current parameters: y_deg={y_deg.item():.6f}, pitch={pitch.item():.6f}, "
                  f"roll={roll.item():.6f}, z={translate_z.item():.6f}, "
                  f"y={translate_y.item():.6f}, x={translate_x.item():.6f}")

            # Display images
            # if display_interval and (it % display_interval == 0):
            if 0:
                with torch.no_grad():
                    # 1. L2 loss를 위한 이미지 비교
                    ref_np = color_ref[0].cpu().numpy()
                    opt_np = color_opt[0].cpu().numpy()
                    
                    scl = display_res // opt_np.shape[0]
                    ref_img = np.repeat(np.repeat(ref_np, scl, axis=0), scl, axis=1)
                    opt_img = np.repeat(np.repeat(opt_np, scl, axis=0), scl, axis=1)

                    result_image = np.concatenate([opt_img, ref_img], axis=1)
                    util.display_image(result_image, size=display_res, title=f'Iteration {it} - L2 Comparison')

                    # # 2. 실루엣 마스크 비교
                    # gray1 = 0.299 * color_opt[0, ..., 0] + 0.587 * color_opt[0, ..., 1] + 0.114 * color_opt[0, ..., 2]
                    # gray2 = 0.299 * color_ref[0, ..., 0] + 0.587 * color_ref[0, ..., 1] + 0.114 * color_ref[0, ..., 2]
                    
                    # mask1 = (gray1 > threshold).float()
                    # mask2 = (gray2 > threshold).float()

                    # # 마스크를 시각화하기 위해 3채널로 변환
                    # mask1_vis = torch.stack([mask1, mask1, mask1], dim=-1).cpu().numpy()
                    # mask2_vis = torch.stack([mask2, mask2, mask2], dim=-1).cpu().numpy()

                    # # 이미지 크기 조정
                    # scl = display_res // mask1_vis.shape[0]
                    # mask1_img = np.repeat(np.repeat(mask1_vis, scl, axis=0), scl, axis=1)
                    # mask2_img = np.repeat(np.repeat(mask2_vis, scl, axis=0), scl, axis=1)

                    # # 마스크 이미지 연결
                    # result_image = np.concatenate([mask1_img, mask2_img], axis=1)
                    # util.display_image(result_image, size=display_res, title=f'Iteration {it} - Silhouette Masks')

   
    # print final best results
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value:.6f}")

    # Save final best image
    print("\nSaving final best image...")
    with torch.no_grad():
        color_opt_img = (color_opt_np * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(color_opt_img).save(os.path.join(output_dir, 
            f'FINAL_BEST_color_opt_loss_{best_loss:.6f}_ydeg_{best_params["y_deg"]:.2f}_pitch_{best_params["pitch"]:.2f}_roll_{best_params["roll"]:.2f}_z_{best_params["translate_z"]:.6f}_y_{best_params["translate_y"]:.6f}_x_{best_params["translate_x"]:.6f}.png'))
        pose_log_path = os.path.join(output_dir, f"best_poses_{object_name}.txt")

        # Write pose line (overwrite on first write, then append)
        with open(pose_log_path, "w") as f:
                f.write(
                    f'{best_params["y_deg"]:.6f} {best_params["pitch"]:.6f} {best_params["roll"]:.6f} '
                    f'{best_params["translate_z"]:.6f} {best_params["translate_y"]:.6f} {best_params["translate_x"]:.6f}\n'
                )
    # Done.
    print("Done.")

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Pose optimization example')
    parser.add_argument('--opengl', help='enable OpenGL rendering', action='store_true', default=False)
    parser.add_argument('--outdir', help='specify output directory', default='')
    parser.add_argument('--mip', help='enable mipmapping', action='store_true', default=False)
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--max-iter', type=int, default=10000)
    parser.add_argument("--object_name", type=str, required=True, help="Subdirectory under obj_data (e.g., book, chair, etc.)")
    parser.add_argument("--y_deg_base", type=float, required=True, help="y_deg_base")
    parser.add_argument("--pitch_base", type=float, required=True, help="pitch_base")
    parser.add_argument("--roll_base", type=float, required=True, help="roll_base")
    parser.add_argument("--translate_z_base", type=float, required=True, help="translate_z_base")
    parser.add_argument("--translate_y_base", type=float, required=True, help="translate_y_base")
    parser.add_argument("--translate_x_base", type=float, required=True, help="translate_x_base")
    args = parser.parse_args()

    # Set up logging.
    if args.outdir:
        ms = 'mip' if args.mip else 'nomip'
        out_dir = f'{args.outdir}/earth_{ms}'
        print (f'Saving results under {out_dir}')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')

    fit_pose(object_name=args.object_name,
             y_deg_base = args.y_deg_base,
             pitch_base = args.pitch_base,
             roll_base = args.roll_base,
             translate_z_base = args.translate_z_base,
             translate_y_base = args.translate_y_base,
             translate_x_base = args.translate_x_base,
             max_iter=args.max_iter, 
             log_interval=10, 
             display_interval=args.display_interval, 
             enable_mip=args.mip, 
             out_dir=out_dir, 
             log_fn='log.txt', 
             use_opengl=args.opengl)

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
