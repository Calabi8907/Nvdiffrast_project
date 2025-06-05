import argparse
import os
import pathlib
import numpy as np
import shutil
from pygltflib import GLTF2

def restore_axis(pos, glb_path):
    gltf = GLTF2().load(glb_path)
    generator = gltf.asset.generator.lower() if gltf.asset and gltf.asset.generator else ""
    if "trimesh" in generator:
        x = pos[:, 0].copy()
        y = pos[:, 1].copy()
        z = pos[:, 2].copy()
        pos[:, 0] = x
        pos[:, 1] = y
        pos[:, 2] = z
        # pass
    elif "blender" in generator:
        x = pos[:, 0].copy()
        y = pos[:, 1].copy()
        z = pos[:, 2].copy()
        pos[:, 0] = x
        pos[:, 1] = -z
        pos[:, 2] = -y

    return pos

def save_obj_with_uv_and_texture(
    pos_path, pos_idx_path, uv_path, uv_idx_path, tex_path, output_dir, glb_path=None
):
    os.makedirs(output_dir, exist_ok=True)
    output_obj_path = os.path.join(output_dir, "output.obj")
    output_mtl_path = os.path.join(output_dir, "output.mtl")
    tex_filename = "tex.png"

    # Load npy files
    pos = np.load(pos_path)
    if glb_path is not None:
        pos = restore_axis(pos, glb_path)
    pos_idx = np.load(pos_idx_path)
    uv = np.load(uv_path)
    uv_idx = np.load(uv_idx_path)

    # Write MTL file
    with open(output_mtl_path, "w") as mtl:
        mtl.write("newmtl material0\n")
        mtl.write(f"map_Kd {tex_filename}\n")

    # Write OBJ file
    with open(output_obj_path, "w") as obj:
        obj.write(f"mtllib output.mtl\n")
        obj.write("usemtl material0\n")
        for v in pos:
            obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for vt in uv:
            obj.write(f"vt {vt[0]:.6f} {1.0-vt[1]:.6f}\n")  # V 플립
        for idx, f in enumerate(pos_idx):
            f_v = f + 1
            f_vt = uv_idx[idx] + 1
            obj.write(
                f"f {f_v[0]}/{f_vt[0]} {f_v[1]}/{f_vt[1]} {f_v[2]}/{f_vt[2]}\n"
            )

    shutil.copy(tex_path, os.path.join(output_dir, tex_filename))
    print(f"OBJ 저장: {output_obj_path}")
    print(f"MTL 저장: {output_mtl_path}")
    print(f"Texture 복사: {os.path.join(output_dir, tex_filename)}")

def main():
    parser = argparse.ArgumentParser(description='Earth texture fitting example')
    parser.add_argument("--object_name", type=str, required=True, help="Subdirectory under obj_data (e.g., book, chair, etc.)")
    args = parser.parse_args()

    # ===== 예시 =====
    base_dir = pathlib.Path(__file__).absolute().parents[3]
    datadir = os.path.join(base_dir, "samples", "data", "npy_data", args.object_name)
    glb_dir = os.path.join(base_dir, "samples", "reference_data", "glb_data")
    export_dir = os.path.join(datadir, "obj_export")

    pos_path = os.path.join(datadir, "pos_o.npy")
    pos_idx_path = os.path.join(datadir, "pos_idx_o.npy")
    uv_path = os.path.join(datadir, "uv_o.npy")
    uv_idx_path = os.path.join(datadir, "uv_idx_o.npy")
    tex_path = os.path.join(datadir, "tex_o.png")
    glb_path = os.path.join(glb_dir, args.object_name + ".glb")

    save_obj_with_uv_and_texture(
        pos_path, pos_idx_path, uv_path, uv_idx_path, tex_path, export_dir, glb_path
    )

if __name__ == "__main__":
    main()
