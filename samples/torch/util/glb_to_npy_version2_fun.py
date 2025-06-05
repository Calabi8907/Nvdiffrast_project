import os
import numpy as np
import pathlib
from PIL import Image
from base64 import b64decode
from pygltflib import GLTF2, Image as GLTFImage

def convert_glb_to_npy(glb_path: str, output_dir: str):
    glb_name = os.path.splitext(os.path.basename(glb_path))[0]
    # output_dir = os.path.join(output_base_dir, glb_name)
    os.makedirs(output_dir, exist_ok=True)


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
            uv_idx = tri.copy() if uv.shape[0] == pos.shape[0] else tri.copy()
            found = True
            break
        if found:
            break
    if not found:
        raise RuntimeError("유효한 primitive가 포함된 mesh를 찾을 수 없습니다.")

    # 텍스처 저장
    tex_path = os.path.join(output_dir, 'tex.png')
    selected_image_index = None
    try:
        mat = gltf.materials[0]
        base_color_tex = mat.pbrMetallicRoughness.baseColorTexture
        if base_color_tex and base_color_tex.index is not None:
            selected_image_index = gltf.textures[base_color_tex.index].source
    except:
        pass
    if selected_image_index is None:
        selected_image_index = 0

    image_info: GLTFImage = gltf.images[selected_image_index]
    if image_info.bufferView is not None:
        view = gltf.bufferViews[image_info.bufferView]
        img_bytes = buffer_bytes[view.byteOffset : view.byteOffset + view.byteLength]
        with open(tex_path, 'wb') as f:
            f.write(img_bytes)
    elif image_info.uri and image_info.uri.startswith('data:'):
        img_bytes = b64decode(image_info.uri.split(',')[1])
        with open(tex_path, 'wb') as f:
            f.write(img_bytes)
    else:
        raise ValueError("이미지 정보가 비어 있거나 외부 URI 형식입니다.")

    # 저장
    np.save(os.path.join(output_dir, 'pos.npy'), pos)
    np.save(os.path.join(output_dir, 'pos_idx.npy'), tri)
    np.save(os.path.join(output_dir, 'uv.npy'), uv)
    np.save(os.path.join(output_dir, 'uv_idx.npy'), uv_idx)

    print(f"[✓] GLB 변환 완료: {output_dir}")
    return output_dir
