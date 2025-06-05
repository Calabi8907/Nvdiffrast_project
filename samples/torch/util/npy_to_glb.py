



import os
import numpy as np
from pygltflib import (
    GLTF2, Buffer, BufferView, Accessor, Primitive, Mesh, Node, Scene,
    Image as GLTFImage, Texture, TextureInfo, Material, PbrMetallicRoughness, Sampler
)

# ====== 사용자 설정 ======
# 복원된 .glb 파일을 저장할 디렉토리
datadir = r'C:\Users\samsung\test\nvdiffrast\samples\torch\samples\obj_data'
# output_glb = os.path.join(datadir, 'restored_model.glb')
output_glb = r'C:\Users\samsung\test\nvdiffrast\samples\torch\samples\glb_data\restored_model.glb'

# ====== NPY 및 PNG 파일 로드 ======
pos_path     = os.path.join(datadir, 'pos_o.npy')
pos_idx_path = os.path.join(datadir, 'pos_idx_o.npy')
uv_path      = os.path.join(datadir, 'uv_o.npy')
uv_idx_path  = os.path.join(datadir, 'uv_idx_o.npy')
tex_path     = os.path.join(datadir, 'tex_o.png')

pos     = np.load(pos_path).astype(np.float32)
pos_idx = np.load(pos_idx_path)
uv      = np.load(uv_path).astype(np.float32)
uv_idx  = np.load(uv_idx_path)  # 일반적으로 pos_idx와 동일 인덱스 사용
with open(tex_path, 'rb') as f:
    img_bytes = f.read()

# ====== 바이너리 블롭 구성 ======
def align(offset, alignment=4):
    return (offset + alignment - 1) // alignment * alignment

sections = []
offset = 0

# 1) POSITION
pos_bytes = pos.tobytes()
offset = align(offset)
sections.append((pos_bytes, offset))
offset += len(pos_bytes)

# 2) INDICES
if pos_idx.dtype == np.uint16:
    idx_comp = 5123
elif pos_idx.dtype in (np.uint32, np.int32):
    idx_comp = 5125
else:
    raise ValueError(f"지원되지 않는 인덱스 dtype: {pos_idx.dtype}")
idx_bytes = pos_idx.ravel().astype(pos_idx.dtype).tobytes()
offset = align(offset)
sections.append((idx_bytes, offset))
offset += len(idx_bytes)

# 3) TEXCOORD_0
uv_bytes = uv.tobytes()
offset = align(offset)
sections.append((uv_bytes, offset))
offset += len(uv_bytes)

# 4) IMAGE (PNG)
offset = align(offset)
sections.append((img_bytes, offset))
offset += len(img_bytes)

# 전체 버퍼 생성
buffer_blob = bytearray(offset)
for data, off in sections:
    buffer_blob[off:off+len(data)] = data

# ====== glTF 구조 생성 ======
gltf = GLTF2()
# 버퍼 설정
gltf.buffers.append(Buffer(byteLength=len(buffer_blob)))

# 버퍼뷰들 등록
for _, off in sections:
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=off, byteLength=0))

# 어세서 등록
# POSITION (VEC3)
gltf.accessors.append(Accessor(
    bufferView=0, byteOffset=0, componentType=5126,
    count=pos.shape[0], type="VEC3"
))
# INDICES (SCALAR)
gltf.accessors.append(Accessor(
    bufferView=1, byteOffset=0, componentType=idx_comp,
    count=pos_idx.size, type="SCALAR"
))
# TEXCOORD_0 (VEC2)
gltf.accessors.append(Accessor(
    bufferView=2, byteOffset=0, componentType=5126,
    count=uv.shape[0], type="VEC2"
))

# IMAGE 등록
gltf.bufferViews[3].byteLength = len(img_bytes)
gltf.images.append(GLTFImage(bufferView=3, mimeType="image/png"))
# Sampler, Texture, Material
gltf.samplers.append(Sampler())
gltf.textures.append(Texture(sampler=0, source=0))
pbr = PbrMetallicRoughness(baseColorTexture=TextureInfo(index=0))
gltf.materials.append(Material(name="mat", pbrMetallicRoughness=pbr))

# Mesh와 Primitive
primitive = Primitive(
    attributes={"POSITION": 0, "TEXCOORD_0": 2},
    indices=1,
    material=0
)
gltf.meshes.append(Mesh(primitives=[primitive]))
# Node와 Scene
gltf.nodes.append(Node(mesh=0))
gltf.scenes.append(Scene(nodes=[0]))
gltf.scene = 0

# 바이너리 블롭 설정 및 GLB 저장
gltf.set_binary_blob(bytes(buffer_blob))
gltf.save_binary(output_glb)

print(f"GLB 복원 완료: {output_glb}")


