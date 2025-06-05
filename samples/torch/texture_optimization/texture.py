import argparse
import os
import pathlib
import sys
from PIL import Image
import numpy as np
import torch

UTIL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'util')
sys.path.append(UTIL_DIR)

from util import util

import nvdiffrast.torch as dr

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

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def render(glctx,
            mtx,    # 모델뷰프로젝션 행렬
            pos,     # 버텍스 위치
            pos_idx,
            uv,
            uv_idx,
            tex,    # 텍스처 이미지 배열
            resolution, # 해상도
            enable_mip, # 밉맵 사용 여부
            max_mip_level): # 최대 밉맵 레벨
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    # Mask out background
    color = color * torch.clamp(rast_out[..., -1:], 0, 1)
    return color


def make_grid(arr, ncols=2):
    # arr: (N, H, W, C)
    n, h, w, c = arr.shape
    nrows = n // ncols
    assert n == nrows * ncols, "Number of images must be a multiple of ncols"
    return arr.reshape(nrows, ncols, h, w, c).swapaxes(1,2).reshape(h * nrows, w * ncols, c)

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
        pass
    else:
        # 기타 툴: 추가적인 분기가 필요할 수 있음 (여기서는 pass)
        pass

    return pos
#----------------------------------------------------------------------------

def fit_texture(object_name,
                y_deg,
              pitch,
              roll,
              translate_z,
              translate_y,
              translate_x,
              max_iter,
              log_interval      = 10,
              display_interval  = None,
              display_res       = 1024,
              enable_mip        = True,
              res               = 1024,
              ref_res           = 2048,
              lr_base           = 1e-2,
              lr_ramp           = 0.1,
              out_dir           = None,
              log_fn            = None,
              texsave_interval  = None,
              texsave_fn        = None,
              imgsave_interval  = None,
              imgsave_fn        = None,
              use_opengl        = False):

    log_file = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(out_dir + '/' + log_fn, 'wt')
    else:
        imgsave_interval, texsave_interval = None, None


    #디렉토리 설정
    base_dir = pathlib.Path(__file__).absolute().parents[3]
    datadir = os.path.join(base_dir, "samples", "data", "npy_data", object_name)
    output_dir = os.path.join(base_dir, "samples", "torch", "output", object_name)

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
    color_ref = torch.from_numpy(ref_img_np).cuda().unsqueeze(0)

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
    tex_opt = torch.from_numpy(tex_o.astype(np.float32)).cuda().requires_grad_()
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    optimizer    = torch.optim.Adam([tex_opt], lr=lr_base)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

    texloss_avg = []

    for it in range(max_iter + 1):


        proj  = util.projection(x=0.4, n=1.0, f=200.0)

        #탐색한 변수.
        # z -> roll ,   x -> pitch   , y -> yaw
        
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
        translate_matrix[0, 3] = translate_x  # 축 이동
        mvp = torch.matmul(torch.tensor(proj, dtype=torch.float32, device='cuda'), translate_matrix)
        mtx_total = torch.matmul(mvp, q_to_mtx(q_tensor))


        # Compute and log texture loss
        # with torch.no_grad():
        #     texmask = torch.zeros_like(tex)
        #     tr = tex.shape[1]//4
        #     texmask[tr+13:2*tr-13, 25:-25, :] += 1.0
        #     texmask[25:-25, tr+13:2*tr-13, :] += 1.0
        #     texloss = (torch.sum(texmask * (tex - tex_opt)**2)/torch.sum(texmask))**0.5
        #     texloss_avg.append(float(texloss))

        # Render reference and optimized
        color_opt = render(glctx, mtx_total, vtx_pos, pos_idx, vtx_uv, uv_idx, tex_opt, res, enable_mip, max_mip_level)

        if it == 1:
            # Show/save image
            color_opt_np = color_opt[0].detach().cpu().numpy()  # OK!

            # Save color_opt (rendered image)
            color_opt_img = (color_opt_np * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(color_opt_img).save(os.path.join(output_dir, f'origin_tex.png'))
            # 한 번만 저장 (원하면)

        # Downsample reference to match res
        while color_ref.shape[1] > res:
            color_ref = util.bilinear_downsample(color_ref)

        if it == 1:
            color_ref_img = (color_ref[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(color_ref_img).save(os.path.join(output_dir, 'color_ref_downsampled.png'))

        # Backprop
        loss = torch.mean((color_ref - color_opt)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 현재 loss 계산안되고 있음.
        # Logging
        if log_interval and (it % log_interval == 0):
            texloss_val = np.mean(np.asarray(texloss_avg))
            texloss_avg = []
            psnr = -10.0 * np.log10(texloss_val**2)
            s = f"iter={it},loss={texloss_val:f},psnr={psnr:f}"
            print(s)
            if log_file:
                log_file.write(s + '\n')
        
        display_image = display_interval and (it % display_interval == 0)
        if display_image:
            with torch.no_grad():
                    # # 1. l2 loss 구할 때 사용한 두 이미지 출력
                    while color_ref.shape[1] > color_opt.shape[1]:
                        color_ref = util.bilinear_downsample(color_ref)

                    ref_np = color_ref[0].cpu().numpy()
                    opt_np = color_opt[0].cpu().numpy()
                    
                    scl = display_res // opt_np.shape[0]
                    ref_img = np.repeat(np.repeat(ref_np, scl, axis=0), scl, axis=1)
                    opt_img = np.repeat(np.repeat(opt_np, scl, axis=0), scl, axis=1)

                    result_image = np.concatenate([opt_img, ref_img], axis=1)
                    util.display_image(result_image, size=display_res, title=f"Y={y_deg}°")

        
    # Show/save image
    color_opt_np = color_opt[0].detach().cpu().numpy()  # OK!

    # Save color_opt (rendered image)
    color_opt_img = (color_opt_np * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(color_opt_img).save(os.path.join(output_dir, f'result_tex.png'))

    if log_file:
        log_file.close()

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
    parser.add_argument("--y_deg", type=float, required=True, help="y_deg")
    parser.add_argument("--pitch", type=float, required=True, help="pitch")
    parser.add_argument("--roll", type=float, required=True, help="roll")
    parser.add_argument("--translate_z", type=float, required=True, help="translate_z")
    parser.add_argument("--translate_y", type=float, required=True, help="translate_y")
    parser.add_argument("--translate_x", type=float, required=True, help="translate_x")
    args = parser.parse_args()

    if args.outdir:
        ms = 'mip' if args.mip else 'nomip'
        out_dir = f'{args.outdir}/earth_{ms}'
        print(f'Saving results under {out_dir}')
    else:
        out_dir = None
        print('No output directory specified, not saving log or images')

    fit_texture(object_name = args.object_name,
                y_deg = args.y_deg,
              pitch = args.pitch,
              roll = args.roll,
              translate_z = args.translate_z,
              translate_y = args.translate_y,
              translate_x = args.translate_x,
              max_iter=args.max_iter,
              log_interval=10,
              display_interval=args.display_interval,
              enable_mip=args.mip,
              out_dir=out_dir,
              log_fn='log.txt',
              texsave_interval=1000,
              texsave_fn='tex_%06d.png',
              imgsave_interval=1000,
              imgsave_fn='img_%06d.png',
              use_opengl=args.opengl)

    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
