from rembg import remove
from PIL import Image
import io
import argparse
import os
import pathlib

def process_image(input_image_path, output_image_path, output_size=(2048, 2048), scale_factor=0.8):
    """
    이미지의 배경을 제거하고 지정된 크기로 조정하여 중앙에 배치한 후 저장합니다.

    Args:
        input_image_path (str): 입력 이미지 파일 경로
        output_image_path (str): 출력 이미지 저장 경로
        output_size (tuple): 출력 이미지 크기 (기본값: (2048, 2048))
        scale_factor (float): 객체 크기 조정 비율 (기본값: 0.8)
    """
    # 이미지 열기
    with open(input_image_path, 'rb') as i:
        input_image = i.read()

    # 배경 제거
    output_image = remove(input_image)

    # PIL Image로 변환
    img = Image.open(io.BytesIO(output_image))

    # 이미지의 알파 채널(투명도)을 기준으로 객체의 경계 상자 찾기
    bbox = img.getbbox()

    # 새로운 이미지 생성 (검정 배경)
    new_img = Image.new('RGB', output_size, (0, 0, 0))

    # 객체 크기 계산
    obj_width = bbox[2] - bbox[0]
    obj_height = bbox[3] - bbox[1]

    # 크기 조정 비율 계산
    target_size = int(output_size[0] * scale_factor)
    scale = min(target_size / obj_width, target_size / obj_height)
    new_width = int(obj_width * scale)
    new_height = int(obj_height * scale)

    # 객체 잘라내고 리사이즈
    resized_obj = img.crop(bbox).resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 중앙 좌표 계산 후 붙여넣기
    x = (output_size[0] - new_width) // 2
    y = (output_size[1] - new_height) // 2
    new_img.paste(resized_obj, (x, y), resized_obj)

    # 결과 저장
    new_img.save(output_image_path)
    print("배경 제거 및 중앙 정렬 완료. 결과는", output_image_path, "에 저장되었습니다.")

# -------------------

def main():
    parser = argparse.ArgumentParser(description='이미지 배경 제거')
    parser.add_argument('--image_name', type=str, default='apple_table.jpg',
                        help='배경을 제거할 이미지 파일명')
    args = parser.parse_args()

    # 경로 설정
    base_dir = pathlib.Path(__file__).absolute().parents[3]
    input_dir = os.path.join(base_dir, "samples", "reference_data", "png_raw")
    output_dir = os.path.join(base_dir, "samples", "reference_data", "png_seg")

    base_name = args.image_name.rsplit('.', 1)[0]
    input_path = os.path.join(input_dir, args.image_name)
    output_path = os.path.join(output_dir, f"{base_name}_seg.png")

    process_image(input_path, output_path)

if __name__ == "__main__":
    main()
