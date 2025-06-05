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
import util

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
    # r0 = torch.stack([1.0-2.0*q[1]**2 - 2.0*q[2]**2, 2.0*q[0]*q[1] - 2.0*q[2]*q[3], 2.0*q[0]*q[2] + 2.0*q[1]*q[3]])
    # r1 = torch.stack([2.0*q[0]*q[1] + 2.0*q[2]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[2]**2, 2.0*q[1]*q[2] - 2.0*q[0]*q[3]])
    # r2 = torch.stack([2.0*q[0]*q[2] - 2.0*q[1]*q[3], 2.0*q[1]*q[2] + 2.0*q[0]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[1]**2])
    # rr = torch.transpose(torch.stack([r0, r1, r2]), 1, 0)
    # rr = torch.cat([rr, torch.tensor([[0], [0], [0]], dtype=torch.float32).cuda()], dim=1) # Pad right column.
    # rr = torch.cat([rr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()], dim=0)  # Pad bottom row.
    # return rr
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



def fit_earth(object_name,
              max_iter          = 20000,
              log_interval      = 10,
              display_interval  = None,
              display_res       = 512,
              enable_mip        = True,
              res               = 512,
              ref_res           = 2048,  # Dropped from 4096 to 2048 to allow using the Cuda rasterizer.
              lr_base           = 1e-2,
              lr_ramp          = 0.1,
              nr_base          = 1.0,        # 노이즈 관련 파라미터 추가
              nr_falloff       = 1e-4,       # 노이즈 관련 파라미터 추가
              grad_phase_start = 0.5,        # 그래디언트 페이즈 시작 시점
              out_dir           = None,
              log_fn            = None,
              texsave_interval  = None,
              texsave_fn        = None,
              imgsave_interval  = None,
              imgsave_fn        = None,
              use_opengl        = False):

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
    output_dir = os.path.join(base_dir, "samples", "torch", "output", object_name)
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


# 쿼터니언 텐서 생성
    pose_target = torch.tensor(q_rnd(), device='cuda')
    # pose_init   = q_rnd()
    # pose_opt    = torch.tensor(pose_init / np.sum(pose_init**2)**0.5, dtype=torch.float32, device='cuda', requires_grad=True)


# best pose 로직
    # fov 관련 로직직
    # # fov_deg = 120.0
    # # fov_rad = math.radians(fov_deg)
    # # compute_fov_y : 등가 초점거리(이미지에 있는 메타데이터) , 센서 width, 센서 hegith
    # # 아이폰 13pro 카메라 spec
    # #                       와이드 카메라 : 7.66 , 5.75
    # #       울트라 와이드 카메라,망원 카메라 : 4.03, 3.02
    # fov_rad = math.radians(compute_fov_y(26,7.66,5.75))
    # # 종횡비 (렌더링 해상도 기준): 예시로 1.0 (정사각형 ref 이미지에 맞춤)
    # aspect_ratio = 1.0
    # # Projection matrix의 x값 계산
    # x = math.tan(fov_rad / 2.0)
    # # util.projection() 호출
    # print(x)

    # proj = util.projection(x=x, n=1.0, f=50.0)
    proj = util.projection(x=0.4, n=1.0, f=200.0)
    # mvp = torch.tensor(np.matmul(proj, util.translate(0, 0, -3.5)).astype(np.float32), device='cuda')

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

    def coarse_y_search_with_xz_noise(render_fn,
                                    loss_fn,
                                    vtx_pos,
                                    pos_idx,
                                    vtx_uv,
                                    uv_idx,
                                    tex_opt,
                                    img_tensor,
                                    # mvp,
                                    glctx,
                                    res,
                                    enable_mip,
                                    max_mip_level,
                                    object_name,
                                    display_interval=1,
                                    display_res=512,
                                    search_iters=10
                                    ):


        best_loss = float('inf')
        best_pose = None
        pose_log_initialized = False

        color_ref = img_tensor
        while color_ref.shape[1] > res:
            color_ref = util.bilinear_downsample(color_ref)

        # 한 번만 저장 (원하면)
        color_ref_img = (color_ref[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(color_ref_img).save(os.path.join(output_dir, 'color_ref_downsampled.png'))

        for y_deg in range(-10,10,1):
            for roll in range(-3,4,1):
                for pitch in range(-10, 10,1):
                    for translate_z in np.arange(1.8, 2.1, 0.05):
                        for translate_y in np.arange(-0.03, 0.04, 0.01):
                            for translate_x in np.arange(-0.03, 0.04, 0.01):

                                #계산
                                ry = np.radians(y_deg)
                                rz = np.radians(roll)
                                rx = np.radians(pitch)
                                q_xyz = euler_xyz_to_quaternion(rx, ry, rz)

                                q_combined = q_xyz
                                q_combined /= np.linalg.norm(q_combined)
                                q_tensor = torch.tensor(q_combined, device='cuda', dtype=torch.float32)

                                translate_matrix = torch.eye(4, dtype=torch.float32, device='cuda')
                                translate_matrix[2, 3] = -3.5 + translate_z  # z축 이동
                                translate_matrix[1, 3] = translate_y  # y축 이동
                                translate_matrix[0, 3] = translate_x # x축 이동
                                mvp = torch.matmul(torch.tensor(proj, dtype=torch.float32, device='cuda'), translate_matrix)
                                mtx_total = torch.matmul(mvp, q_to_mtx(q_tensor))


                                color_opt = render_fn(glctx, mtx_total, vtx_pos, pos_idx, vtx_uv, uv_idx,
                                                    tex_opt, res, enable_mip, max_mip_level)
                                

                                # 손실 계산
                                # loss_fn = PerceptualLoss().cuda()
                                # loss = loss_fn(color_opt, color_ref)
                                # loss = combined_loss(color_opt, color_ref, mode='ssim')   # L2 + SSIM
                                # 또는
                                # loss = combined_loss(color_opt, color_ref, mode='grad')   # L2 + Gradient
                                diff = (color_opt - color_ref) ** 2
                                diff = torch.tanh(5.0 * torch.max(diff, dim=-1)[0])
                                l2_loss = torch.mean(diff)
                                threshold = 0.01
                                sil_loss = silhouette_loss(color_opt, color_ref, threshold)
                                loss = 0.3 * l2_loss + 0.7 * sil_loss
                                # loss = sil_loss



                                # 최적의 포즈 저장
                                loss_val = float(loss)
                                if (loss_val < best_loss) and (loss_val > 0.0):
                                    best_loss = loss_val
                                    best_pose = q_tensor.detach().clone()
                                    # save file
                                    with torch.no_grad():
                                        # Convert tensors to numpy arrays and save
                                        color_opt_np = color_opt[0].cpu().numpy()
                                        # Save color_opt (rendered image)
                                        color_opt_img = (color_opt_np * 255.0).clip(0, 255).astype(np.uint8)
                                        Image.fromarray(color_opt_img).save(os.path.join(output_dir, 
                                        f'loss_{loss_val:.6f}_color_opt_ydeg_{y_deg:03d}_pitch_{pitch:03d}_roll_{roll:03d}_z_{translate_z:.1f}_y_{translate_y:.2f}_x_{translate_x:.2f}.png'))
                                        # Prepare pose log path
                                        pose_log_path = os.path.join(output_dir, f"best_poses_{object_name}.txt")

                                        # Write pose line (overwrite on first write, then append)
                                        mode = "w" if not pose_log_initialized else "a"
                                        with open(pose_log_path, mode) as f:
                                            pose_np = best_pose.cpu().numpy()
                                            pose_str = " ".join([f"{v:.6f}" for v in pose_np])
                                            f.write(f"y_deg={y_deg}, pitch={pitch}, roll={roll}, z={translate_z:.1f},y={translate_y:.1f},x={translate_x:.1f}, loss={loss_val:.6f}, pose=[{pose_str}]\n")

                                        pose_log_initialized = True  # 이후부터는 append

                                # Print/save log.
                                if 1:
                                    s = ("loss=%f,loss_best=%f,best_pose=%s,y=%2f,p=%2f,r=%2f,z=%2f,y=%2f,x=%2f" % 
                                        (loss_val, best_loss, best_pose.cpu().numpy(), y_deg, pitch, roll, translate_z, translate_y, translate_x))
                                    print(s)
                                    if log_file:
                                        log_file.write(s + "\n")
                                        
                                #🔍 Display image every N degrees
                                if 0:
                                    with torch.no_grad():
                                        # # 1. l2 loss 구할 때 사용한 두 이미지 출력
                                        # color_ref = img_tensor
                                        # while color_ref.shape[1] > color_opt.shape[1]:
                                        #     color_ref = util.bilinear_downsample(color_ref)

                                        # ref_np = color_ref[0].cpu().numpy()
                                        # opt_np = color_opt[0].cpu().numpy()
                                        
                                        # scl = display_res // opt_np.shape[0]
                                        # ref_img = np.repeat(np.repeat(ref_np, scl, axis=0), scl, axis=1)
                                        # opt_img = np.repeat(np.repeat(opt_np, scl, axis=0), scl, axis=1)

                                        # result_image = np.concatenate([opt_img, ref_img], axis=1)
                                        # util.display_image(result_image, size=display_res, title=f"Y={y_deg}°")

                                        # 2. 실루엣 loss 구할 때 사용한 두 이미지 출력
                                        # 실루엣 마스크 계산
                                        gray1 = 0.299 * color_opt[0, ..., 0] + 0.587 * color_opt[0, ..., 1] + 0.114 * color_opt[0, ..., 2]
                                        gray2 = 0.299 * color_ref[0, ..., 0] + 0.587 * color_ref[0, ..., 1] + 0.114 * color_ref[0, ..., 2]
                                        
                                        mask1 = (gray1 > threshold).float()
                                        mask2 = (gray2 > threshold).float()

                                        # 마스크를 시각화하기 위해 3채널로 변환
                                        mask1_vis = torch.stack([mask1, mask1, mask1], dim=-1).cpu().numpy()
                                        mask2_vis = torch.stack([mask2, mask2, mask2], dim=-1).cpu().numpy()

                                        # 이미지 크기 조정
                                        scl = display_res // mask1_vis.shape[0]
                                        mask1_img = np.repeat(np.repeat(mask1_vis, scl, axis=0), scl, axis=1)
                                        mask2_img = np.repeat(np.repeat(mask2_vis, scl, axis=0), scl, axis=1)

                                        # 마스크 이미지 연결
                                        result_image = np.concatenate([mask1_img, mask2_img], axis=1)
                                        util.display_image(result_image, size=display_res, title=f"Y={y_deg}° - Silhouette Masks")

        return best_pose, best_loss
    # 제한된 각도록 쿼터니언 기반 포즈 최적화 설정
    pose_opt, loss_best = coarse_y_search_with_xz_noise(
        render_fn=render,
        loss_fn=lambda out, ref: torch.mean(torch.tanh(5.0 * torch.max((out - ref)**2, dim=-1)[0])),
        vtx_pos=vtx_pos,
        pos_idx=pos_idx,
        vtx_uv=vtx_uv,
        uv_idx=uv_idx,
        tex_opt=tex_opt,
        img_tensor=img_tensor,
        # mvp=mvp,
        glctx=glctx,
        res=res,
        enable_mip=enable_mip,
        max_mip_level=max_mip_level,
        object_name = object_name,
        display_interval=30,
        display_res=512,
        search_iters=10
    )





# 최적화 관련 설정정
#     pose_opt.requires_grad_()
#     # Add translate_z as optimization parameter
#     translate_z = torch.tensor(0.0, device='cuda', requires_grad=True)
#     pose_best   = pose_opt.detach().clone()

# # 최적화 코드
#     # Adam optimizer for texture with a learning rate ramp.
#     # optimizer    = torch.optim.Adam([tex_opt], lr=lr_base)
#     optimizer = torch.optim.Adam([pose_opt], betas=(0.9, 0.999), lr=lr_base)
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

#     # Create output directory
#     output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
#     os.makedirs(output_dir, exist_ok=True)

#     # Render.
    
#     texloss_avg = []
#     for it in range(max_iter + 1):
#         # 학습률 설정
#         itf = 1.0 * it / max_iter
#         nr = nr_base * nr_falloff**itf
#         # lr = lr_base * lr_ramp**itf
#         # lr = max(lr_base * lr_ramp**(it / max_iter), 1e-3)
#         lr_base = 1e-1
#         lr_ramp = 0.2       # 10%로 점점 감소
#         lr_min  = 1e-4      # 최소 학습률

#         # 현재 iteration 비율
#         itf = it / max_iter

#         # 스케줄 적용
#         lr = max(lr_base * (lr_ramp ** itf), lr_min)


#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

        
#         # 노이즈 추가
#         # if itf >= grad_phase_start:
#         #     noise = q_unit()
        
#         noise = q_unit()

#         # # 렌더링 행렬 계산 - 원래 코드 부분이나, best pose 로직을 위해 위로 이동동
#         # proj = util.projection(x=0.4, n=1.0, f=200.0)
#         # mvp = torch.tensor(np.matmul(proj, util.translate(0, 0, -3.5)).astype(np.float32), device='cuda')


#         # 목표 이미지와 최적화된 이미지 렌더링
#         color = img_tensor
#         pose_total_opt = q_mul_torch(pose_opt, noise)

#         flip_x = torch.diag(torch.tensor([-1, 1, 1, 1], dtype=torch.float32, device='cuda'))
#         mtx_total_opt = torch.matmul(mvp, torch.matmul(flip_x, q_to_mtx(pose_total_opt)))
#         # mtx_total_opt = torch.matmul(mvp, q_to_mtx(pose_total_opt))
#         color_opt = render(glctx, mtx_total_opt, vtx_pos, pos_idx, vtx_uv, uv_idx, tex_opt, res, enable_mip, max_mip_level)

#         # Reduce the reference to correct size.
#         while color.shape[1] > res:
#             color = util.bilinear_downsample(color)


#         # loss_fn = PerceptualLoss().cuda()
#         # 손실 계산
#         # loss = loss_fn(color_opt, color)
#         # loss = combined_loss(color_opt, color, mode='ssim')   # L2 + SSIM
#         # 또는
#         # loss = combined_loss(color_opt, color, mode='grad')   # L2 + Gradient

#         # 손실 계산 및 최적화
#         diff = (color_opt - color)**2 # L2 norm.
#         diff = torch.tanh(5.0 * torch.max(diff, dim=-1)[0])
#         loss = torch.mean(diff)

#         loss_val = float(loss)

#         # # 최적의 포즈 저장
#         # if (loss_val < loss_best) and (loss_val > 0.0):
#         #     pose_best = pose_total_opt.detach().clone()
#         #     loss_best = loss_val
#         #     if itf < grad_phase_start:
#         #         with torch.no_grad(): pose_opt[:] = pose_best

#         # Best pose 저장 or rollback
#         if loss_val < loss_best:
#             loss_best = loss_val
#             pose_best = pose_opt.detach().clone()
#         else:
#             with torch.no_grad():
#                 pose_opt[:] = pose_best  # rollback

#         # 그래디언트 단계
#         # if itf >= grad_phase_start:
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # 쿼터니언 정규화
#         with torch.no_grad():
#             pose_opt /= torch.sum(pose_opt**2)**0.5

#         # Print/save log.
#         if log_interval and (it % log_interval == 0):
#             err = q_angle_deg(pose_opt, pose_target)
#             ebest = q_angle_deg(pose_best, pose_target)
#             s = "iter=%d,err=%f,err_best=%f,loss=%f,loss_best=%f,lr=%f,nr=%f" % (it, err, ebest, loss_val, loss_best, lr, nr)
#             print(s)
#             if log_file:
#                 log_file.write(s + "\n")

#         # Show/save image.
#         display_image = display_interval and (it % display_interval == 0)
#         save_image = imgsave_interval and (it % imgsave_interval == 0)
#         save_texture = texsave_interval and (it % texsave_interval) == 0

#         if display_image or save_image:

#             with torch.no_grad():

#                 img_b = color[0].cpu().numpy()
#                 img_o = color_opt[0].cpu().numpy()

#                 scl = display_res // img_o.shape[0]
#                 img_b = np.repeat(np.repeat(img_b, scl, axis=0), scl, axis=1)
#                 img_o = np.repeat(np.repeat(img_o, scl, axis=0), scl, axis=1)
#                 result_image = make_grid(np.stack([img_o, img_b]))

#                 if display_image:
#                     util.display_image(result_image, size=display_res, title='%d / %d' % (it, max_iter))
#                 if save_image:
#                     util.save_image(out_dir + '/' + (imgsave_fn % it), result_image)

#                 if save_texture:
#                     texture = tex_opt.cpu().numpy()[::-1]
#                     util.save_image(out_dir + '/' + (texsave_fn % it), texture)

    # save file
    # 저장할 텐서 이름과 변수 매핑
    to_save = {
        "pos_idx": pos_idx,
        "pos":     vtx_pos,
        "uv_idx":  uv_idx,
        "uv":      vtx_uv,
        "tex":     tex_opt,      # 최적화된 텍스처
    }
    for name, tensor in to_save.items():
        arr = tensor.detach().cpu().numpy()
        if name == "tex":
            # arr shape: (H, W, 3), 값 범위 [0,1] 인 float32
            img_arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
            out_path = os.path.join(datadir, f"{name}_o.png")
            img.save(out_path)
        else:
            out_path = os.path.join(datadir, f"{name}_o.npy")
            np.save(out_path, arr)
        print(f"Saved {out_path}")
    
    # Done.
    if log_file:
        log_file.close()

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Earth texture fitting example')
    parser.add_argument('--opengl', help='enable OpenGL rendering', action='store_true', default=False)
    parser.add_argument('--outdir', help='specify output directory', default='')
    parser.add_argument('--mip', help='enable mipmapping', action='store_true', default=False)
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--max-iter', type=int, default=10000)
    parser.add_argument("--object_name", type=str, required=True, help="Subdirectory under obj_data (e.g., book, chair, etc.)")
    args = parser.parse_args()

    # Set up logging.
    if args.outdir:
        ms = 'mip' if args.mip else 'nomip'
        out_dir = f'{args.outdir}/earth_{ms}'
        print (f'Saving results under {out_dir}')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')

    # Run.
    fit_earth(object_name = args.object_name, max_iter=args.max_iter, log_interval=10, display_interval=args.display_interval, enable_mip=args.mip, out_dir=out_dir, log_fn='log.txt', texsave_interval=1000, texsave_fn='tex_%06d.png', imgsave_interval=1000, imgsave_fn='img_%06d.png', use_opengl=args.opengl)

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
