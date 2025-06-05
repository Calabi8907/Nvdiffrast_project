from pygltflib import GLTF2
import sys

def inspect_glb(glb_path):
    gltf = GLTF2().load(glb_path)

    print(f"✅ Loaded GLB: {glb_path}")
    print(f"Scenes: {len(gltf.scenes)}")
    print(f"Nodes: {len(gltf.nodes)}")
    print(f"Meshes: {len(gltf.meshes)}")
    print()

    # Scene structure
    for i, scene in enumerate(gltf.scenes):
        print(f"Scene {i}: name={scene.name}, nodes={scene.nodes}")

    # Check root node transforms
    for i, node in enumerate(gltf.nodes):
        print(f"Node {i}: name={node.name}")
        if node.translation:
            print(f"  └─ translation: {node.translation}")
        if node.rotation:
            print(f"  └─ rotation: {node.rotation}")
        if node.scale:
            print(f"  └─ scale: {node.scale}")
        if node.matrix:
            print(f"  └─ matrix: {node.matrix}")
        if node.mesh is not None:
            print(f"  └─ mesh: {node.mesh}")
        print()

    # Check asset metadata
    if gltf.asset:
        print("Asset Metadata:")
        print(f"  Generator: {gltf.asset.generator}")
        print(f"  Version: {gltf.asset.version}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_glb_metadata.py path_to_model.glb")
        sys.exit(1)
    
    glb_file = sys.argv[1]
    inspect_glb(glb_file)
