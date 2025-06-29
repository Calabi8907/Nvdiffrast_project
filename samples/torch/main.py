# main.py
import argparse
import os
import pathlib
from pose_optimization.pose_gd import fit_pose
from texture_optimization.texture import fit_texture
from util.npy_to_obj import save_obj_with_uv_and_texture
from util.glb_to_npy_version2_fun import convert_glb_to_npy
from util.util import close_window
from image_segmentation import process_image

# 실행 아직 안해봄. ******************************************

def main():
    parser = argparse.ArgumentParser(description='Pose optimization example')
    parser.add_argument("--object_name", type=str, required=True, help="Subdirectory under obj_data (e.g., book, chair, etc.)")
    parser.add_argument("--y_deg_base", type=float, required=True, help="y_deg_base")
    parser.add_argument("--pitch_base", type=float, required=True, help="pitch_base")
    parser.add_argument("--roll_base", type=float, required=True, help="roll_base")
    parser.add_argument("--translate_z_base", type=float, required=True, help="translate_z_base")
    parser.add_argument("--translate_y_base", type=float, required=True, help="translate_y_base")
    parser.add_argument("--translate_x_base", type=float, required=True, help="translate_x_base")
    args = parser.parse_args()
    
    # 객체 이름 지정 (예: box, book, mug 등)
    # 기본 설정값
    max_iter = 1000
    log_interval = 10
    display_interval = 1
    enable_mip = True
    use_opengl = False

    # 데이터 경로 베이스스
    base_dir = pathlib.Path(__file__).absolute().parents[2]

    # 입력 이미지 관련 저장 경로로
    image_input_path = os.path.join(base_dir, "samples", "reference_data", "png_raw")
    image_input = os.path.join(image_input_path, args.object_name + ".png")
    image_seg_output_path = os.path.join(base_dir, "samples", "reference_data", "png_seg")
    image_seg_output = os.path.join(image_seg_output_path, f"{args.object_name}_seg.png")

    # npy 파일 경로 및 export 경로로
    # datadir = os.path.join(base_dir, "samples", "data", "npy_data", args.object_name)
    # export_dir = os.path.join(datadir, "obj_export")
    datadir = base_dir / "samples" / "data" / "npy_data" / args.object_name
    export_dir = datadir / "obj_export"
    

    # glb 파일 참조 경로 
    glb_dir = os.path.join(base_dir, "samples", "reference_data", "glb_data")
    glb_path = os.path.join(glb_dir, args.object_name + ".glb")
    glb_path = base_dir / "samples" / "reference_data" / "glb_data" / (args.object_name + ".glb")
    npy_output_dir = os.path.join(base_dir, "samples", "data", "npy_data", args.object_name)
    
    # 그라디언트 결과 저장 경로 (pose)
    output_dir = os.path.join(base_dir, "samples", "torch", "output", args.object_name, "output_pose_gd")
    pose_log_path = os.path.join(output_dir, f"best_poses_{args.object_name}.txt")


    # 0. Preprocessing
    convert_glb_to_npy(str(glb_path), str(npy_output_dir))          # 3D data
    process_image(image_input, image_seg_output)                    # Image segmentation

    # 1. Pose Optimization
    print("\n[Step 1] Pose Optimization 시작...")
    print(f"Running pose optimization for: {args.object_name}")
    fit_pose(
        object_name=args.object_name,
        y_deg_base = args.y_deg_base,
        pitch_base = args.pitch_base,
        roll_base = args.roll_base,
        translate_z_base = args.translate_z_base,
        translate_y_base = args.translate_y_base,
        translate_x_base = args.translate_x_base,
        max_iter=max_iter,
        log_interval=log_interval,
        display_interval=display_interval,
        enable_mip=enable_mip,
        out_dir=None,
        log_fn='log.txt',
        use_opengl=use_opengl
    )
    
    with open(pose_log_path, "r") as f:
        line = f.readline()
        y_deg, pitch, roll, translate_z, translate_y, translate_x = map(float, line.strip().split())
    
    print(f"best_pose_param = y_deg: {y_deg}, pitch: {pitch}, roll: {roll}, translate_z: {translate_z}, translate_y: {translate_y}, translate_x: {translate_x}")
    close_window()
    # 2. Texture Optimization
    print("\n[Step 2] Texture Optimization 시작...")
    fit_texture(
        object_name=args.object_name,
        y_deg = y_deg,
        pitch = pitch,  
        roll = roll,
        translate_z = translate_z,
        translate_y = translate_y,
        translate_x = translate_x,
        max_iter=max_iter,
        log_interval=log_interval,
        display_interval=display_interval,
        display_res=1024,
        enable_mip=enable_mip,
        res=1024,
        ref_res=2048,
        lr_base=1e-2,
        lr_ramp=0.1,
    )
    close_window()
    # 3. Export to Obj
    print("\n📦 Step 3: Export OBJ")
    save_obj_with_uv_and_texture (
        pos_path=datadir / "pos_o.npy",
        pos_idx_path=datadir / "pos_idx_o.npy",
        uv_path=datadir / "uv_o.npy",
        uv_idx_path=datadir / "uv_idx_o.npy",
        tex_path=datadir / "tex_o.png",
        output_dir=export_dir,
        glb_path=glb_path
    )

    print("\n✅ 전체 프로세스 완료.")

if __name__ == "__main__":
    main()
