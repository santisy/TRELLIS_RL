import os
import copy
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import open3d as o3d
import utils3d

def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects', debug=False) -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    
    metadata = metadata.to_dict('records')
    records = []
    max_workers = max_workers or os.cpu_count()
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                try:
                    local_path = metadatum['local_path']
                    sha256 = metadatum['sha256']
                    file = local_path # THis is the different behavior
                    hdri_name = metadatum.get('hdri_name')
                    
                    if debug:
                        print(f"Processing file: {file}, SHA256: {sha256}")
                        if not os.path.exists(file):
                            print(f"ERROR: File does not exist: {file}")
                    
                    record = func(file, sha256, hdri_name)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}")
        
    return pd.DataFrame.from_records(records)


def _voxelize(file, sha256, hdri_name, output_dir):
    mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'renders', sha256, hdri_name, 'mesh.ply'))
    voxel_out_dir = os.path.join(output_dir, 'voxels', hdri_name)
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    os.makedirs(voxel_out_dir, exist_ok=True)
    utils3d.io.write_ply(os.path.join(voxel_out_dir, f'{sha256}.ply'), vertices)
    return {'sha256': sha256, 'hdri_name': hdri_name, 'voxelized': True, 'num_voxels': len(vertices)}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=None)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'voxels'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' not in metadata.columns:
            raise ValueError('metadata.csv does not have "rendered" column, please run "build_metadata.py" first')
        metadata = metadata[metadata['rendered'] == True]
        if 'voxelized' in metadata.columns:
            metadata = metadata[metadata['voxelized'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')):
            pts = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
            records.append({'sha256': sha256, 'voxelized': True, 'num_voxels': len(pts)})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_voxelize, output_dir=opt.output_dir)
    voxelized = foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Voxelizing')
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    voxelized.to_csv(os.path.join(opt.output_dir, f'voxelized_{opt.rank}.csv'), index=False)
