import os
import math
import numpy as np
import torch
from PIL import Image
import pathlib
import nvdiffrast.torch as dr
import util

def compute_fov_y(focal_length_equiv_mm, sensor_width_mm, sensor_height_mm):
    sensor_diagonal = math.sqrt(sensor_width_mm**2 + sensor_height_mm**2)
    crop_factor = 43.3 / sensor_diagonal
    focal_length_actual = focal_length_equiv_mm / crop_factor
    fov_y_rad = 2 * math.atan(sensor_height_mm / (2 * focal_length_actual))
    return math.degrees(fov_y_rad)

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

    color = color * torch.clamp(rast_out[..., -1:], 0, 1)
    return color

def euler_xyz_to_matrix(rx, ry, rz):
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rx @ Ry @ Rz

def main():
    from itertools import product

    datadir = os.path.join(pathlib.Path(__file__).absolute().parents[0], "samples", "obj_data")
    pos_idx = torch.from_numpy(np.load(os.path.join(datadir, "pos_idx.npy")).astype(np.int32)).cuda()
    pos = torch.from_numpy(np.load(os.path.join(datadir, "pos.npy")).astype(np.float32)).cuda()
    uv_idx = torch.from_numpy(np.load(os.path.join(datadir, "uv_idx.npy")).astype(np.int32)).cuda()
    uv = torch.from_numpy(np.load(os.path.join(datadir, "uv.npy")).astype(np.float32)).cuda()
    tex_img = Image.open(os.path.join(datadir, "tex.png")).convert("RGB")
    tex = torch.from_numpy(np.asarray(tex_img).astype(np.float32) / 255.0).cuda()

    glctx = dr.RasterizeCudaContext()
    res = 512
    enable_mip = True
    max_mip_level = 9

    img_path = r"D:\nvdiffrast\samples\reference_data\tumbler_black_large.png"
    img = Image.open(img_path).convert("RGB").resize((res, res), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).cuda().unsqueeze(0)

    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)

    fov_y = 50.0
    aspect = 1.0
    near, far = 0.1, 100.0
    f = 1.0 / math.tan(math.radians(fov_y) / 2)
    proj = np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

    cam_location_blender = np.array([-1.20149, -2.6687, 2.4388], dtype=np.float32)
    cam_location = np.array([
        cam_location_blender[0],   # X 그대로
        cam_location_blender[2],   # Z → Y
        -cam_location_blender[1]   # Y → -Z
    ], dtype=np.float32)

    cam_rotation_blender = [64.773, 0.0000022, -23.6]
    cam_rotation_rad = np.radians([
        cam_rotation_blender[0],    # X 그대로
        cam_rotation_blender[2],    # Z → Y
        -cam_rotation_blender[1]    # Y → -Z
    ])

    R_cam = euler_xyz_to_matrix(*cam_rotation_rad)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R_cam.T
    view[:3, 3] = -R_cam.T @ cam_location
    mvp = torch.tensor(proj @ view, device='cuda')

    pitch_vals = [0.0, 15.0, 30.0, 45.0, 64.773]
    yaw_vals = [-45.0, -30.0, -23.6, -15.0, 0.0]
    roll_vals = [0.0, 10.0, 20.0]

    idx = 0
    for pitch, yaw, roll in product(pitch_vals, yaw_vals, roll_vals):
        obj_rot_rad = np.radians([pitch, yaw, roll])
        R_obj = euler_xyz_to_matrix(*obj_rot_rad)
        pos_rotated = pos @ torch.tensor(R_obj.T, dtype=torch.float32, device='cuda')

        color = render(glctx, mvp, pos_rotated, pos_idx, uv, uv_idx, tex, res, enable_mip, max_mip_level)
        diff = (color - img_tensor) ** 2
        diff = torch.tanh(5.0 * torch.max(diff, dim=-1)[0])
        loss = torch.mean(diff).item()

        img_np = (color[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        filename = f"obj_rotation_{idx:03d}_p{pitch:.1f}_y{yaw:.1f}_r{roll:.1f}_loss{loss:.4f}.png"
        Image.fromarray(img_np).save(os.path.join(out_dir, filename))
        idx += 1

    print("[디버그] 카메라 위치와 시야 방향:")
    print(f"Camera position (T): {cam_location}")
    print(f"Camera forward vector (look_dir): {R_cam.T @ np.array([0, 0, -1])}")
    print("==============================")

if __name__ == "__main__":
    main()
