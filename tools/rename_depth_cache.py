import argparse
from glob import glob
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename cache from dd3dv2')
    parser.add_argument('--source-dir', type=str, required=True, help='source dir of cache')
    parser.add_argument('--target-dir', type=str, required=True, help='source dir of cache')
    args = parser.parse_args()

    source_dir = args.source_dir
    target_dir = args.target_dir
    os.makedirs(target_dir, exist_ok=True)

    cache_files = sorted(glob(os.path.join(source_dir, "*.npz")), key=lambda x: int(os.path.basename(x).split('_')[1]))
    visited = set()
    for cache_file in tqdm(cache_files):
        cache_name = os.path.basename(cache_file)
        components = cache_name.split('_')
        scene_name, global_idx = components[:2]
        if scene_name not in visited:
            global_start_idx = int(global_idx)
            visited.add(scene_name)

        sample_id = int(global_idx) - int(global_start_idx)
        components[1] = f"{sample_id:03d}"
        cache_name = "_".join(components)
        symlink_path = os.path.join(target_dir, cache_name)
        os.system(f'ln -s {cache_file} {symlink_path}')
