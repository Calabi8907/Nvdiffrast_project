import os
import sys
import numpy as np
import pathlib
from PIL import Image
from base64 import b64decode
from pygltflib import GLTF2, Image as GLTFImage

# ====== 사용자 입력 처리 ======
if len(sys.argv) < 2:
    print("사용법: python glb_to_npy_version2.py [GLB 파일 절대 경로]")
    sys.exit(1)

glb_path = sys.argv[1]
glb_name = os.path.splitext(os.path.basename(glb_path))[0]
base_dir = pathlib.Path(__file__).absolute().parents[3]
output_dir = os.path.join(base_dir, "samples", "data", "npy_data", glb_name)
os.makedirs(output_dir, exist_ok=True)

# ====== GLB 로딩 ======
gltf = GLTF2().load(glb_path)
buffer_bytes = gltf.binary_blob()

def get_data_from_accessor(accessor_id):
    accessor = gltf.accessors[accessor_id]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    dtype_map = {
        ('SCALAR', 5126): np.float32,
        ('VEC2',   5126): np.float32,
        ('VEC3',   5126): np.float32,
        ('VEC4',   5126): np.float32,
        ('SCALAR', 5123): np.uint16,
        ('SCALAR', 5125): np.uint32,
    }
    dtype = dtype_map.get((accessor.type, accessor.componentType))
    if dtype is None:
        raise ValueError(f"지원하지 않는 accessor 타입: {accessor.type}, {accessor.componentType}")

    offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    count = accessor.count
    comps = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[accessor.type]
    byte_length = count * comps * np.dtype(dtype).itemsize

    if offset + byte_length > len(buffer_bytes):
        raise ValueError("GLB 내부 버퍼 범위를 초과했습니다.")

    data = np.frombuffer(buffer_bytes, dtype=dtype, count=count * comps, offset=offset)
    return data.reshape((count, comps))

# ====== Primitive 처리 (첫 mesh 기준) ======
found = False
for mesh in gltf.meshes:
    for primitive in mesh.primitives:
        if not primitive.attributes or primitive.indices is None:
            continue

        pos = get_data_from_accessor(primitive.attributes.POSITION)
        tri = get_data_from_accessor(primitive.indices)

        if tri.shape[1] == 1:
            tri = tri.reshape(-1, 3)
        elif tri.shape[1] != 3:
            raise ValueError(f"삼각형 인덱스가 아닙니다. tri.shape={tri.shape}")

        uv_attr = getattr(primitive.attributes, "TEXCOORD_0", None)
        if uv_attr is not None:
            uv = get_data_from_accessor(uv_attr)
        else:
            raise ValueError("GLB에 TEXCOORD_0 (UV 정보)가 없습니다.")

        if uv.shape[0] == pos.shape[0]:
            uv_idx = tri.copy()
        else:
            print(f"경고: uv.shape[0] != pos.shape[0] → uv 인덱스 분리 필요할 수 있음. 임시로 pos_idx 사용")
            uv_idx = tri.copy()

        found = True
        break
    if found:
        break

if not found:
    raise RuntimeError("유효한 primitive가 포함된 mesh를 찾을 수 없습니다.")

# ====== 텍스처 저장 (baseColorTexture 기준) ======
tex_path = os.path.join(output_dir, 'tex.png')

# 기본은 첫 텍스처 (백업용)
default_image_index = 0
selected_image_index = None

try:
    mat = gltf.materials[0]
    base_color_tex = mat.pbrMetallicRoughness.baseColorTexture
    if base_color_tex and base_color_tex.index is not None:
        selected_image_index = gltf.textures[base_color_tex.index].source
except Exception as e:
    print(f"[경고] baseColorTexture 추출 실패: {e}")

# fallback
if selected_image_index is None:
    print("[경고] baseColorTexture가 명시되지 않았습니다. images[0]을 사용합니다.")
    selected_image_index = default_image_index

image_info: GLTFImage = gltf.images[selected_image_index]

if image_info.bufferView is not None:
    img_view = gltf.bufferViews[image_info.bufferView]
    img_bytes = buffer_bytes[img_view.byteOffset : img_view.byteOffset + img_view.byteLength]
    with open(tex_path, 'wb') as f:
        f.write(img_bytes)
elif image_info.uri:
    if image_info.uri.startswith('data:'):
        base64_data = image_info.uri.split(',')[1]
        img_bytes = b64decode(base64_data)
        with open(tex_path, 'wb') as f:
            f.write(img_bytes)
    else:
        raise ValueError("외부 파일 URI는 지원되지 않습니다. 내부 포함된 data URI만 처리됩니다.")
else:
    raise ValueError("이미지 정보가 비어 있습니다.")

# ====== 저장 ======
np.save(os.path.join(output_dir, 'pos.npy'), pos)
np.save(os.path.join(output_dir, 'pos_idx.npy'), tri)
np.save(os.path.join(output_dir, 'uv.npy'), uv)
np.save(os.path.join(output_dir, 'uv_idx.npy'), uv_idx)

# ====== 요약 출력 ======
print(f"[✓] pos.shape     = {pos.shape}")
print(f"[✓] pos_idx.shape = {tri.shape}")
print(f"[✓] uv.shape      = {uv.shape}")
print(f"[✓] uv_idx.shape  = {uv_idx.shape}")
print(f"[✓] 텍스처 저장 완료: {tex_path}")
print("✅ 변환 완료:", output_dir)
