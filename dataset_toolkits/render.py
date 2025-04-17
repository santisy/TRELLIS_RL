import os
import json
import sys
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence

BLENDER_PATH = os.environ.get("BLENDER_PATH", None)
if BLENDER_PATH is None:
    raise ValueError("Must first specify the Blender binary path as $BLENDER_PATH")
    
def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
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
                    file = os.path.join(output_dir, local_path)
                    hdri_path = metadatum.get('hdri_path')
                    hdri_name = metadatum.get('hdri_name')
                    
                    record = func(file, sha256, hdri_path=hdri_path, hdri_name=hdri_name)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    return pd.DataFrame.from_records(records)

def _render(file_path, sha256, output_dir, num_views, hdri_path=None, hdri_name=None):
    base_folder = os.path.join(output_dir, 'renders', sha256)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--engine', 'CYCLES',
        '--save_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    # Render with hdri if provided
    if hdri_path:
        output_folder = os.path.join(base_folder, hdri_name or os.path.splitext(os.path.basename(hdri_path))[0])
        args_run = args + ['--hdri_path', hdri_path,
                          '--output_folder', output_folder]
        call(args_run, stdout=DEVNULL, stderr=DEVNULL)
        
        if os.path.exists(os.path.join(output_folder, 'transforms.json')):
            return {
                'sha256': sha256,
                'hdri_path': hdri_path,
                'hdri_name': hdri_name or os.path.splitext(os.path.basename(hdri_path))[0],
                'rendered': True
            }
    # No hdri case
    else:
        args += ['--output_folder', base_folder]
        call(args, stdout=DEVNULL, stderr=DEVNULL)
    
        if os.path.exists(os.path.join(base_folder, 'transforms.json')):
            return {
                'sha256': sha256,
                'hdri_path': None,
                'hdri_name': None,
                'rendered': True
            }
    
    return None

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
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)
    
    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    # Filter metadata based on command line arguments
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None and 'aesthetic_score' in metadata.columns:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' in metadata.columns:
            metadata = metadata[metadata['rendered'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # Split metadata for distributed rendering
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # Filter out already rendered object-HDRI combinations
    filtered_metadata = []
    for _, row in metadata.iterrows():
        sha256 = row['sha256']
        hdri_name = row.get('hdri_name')
        
        # Check if this specific object-HDRI combination is already rendered
        if hdri_name:
            output_path = os.path.join(opt.output_dir, 'renders', sha256, hdri_name, 'transforms.json')
        else:
            output_path = os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')
            
        if os.path.exists(output_path):
            records.append({
                'sha256': sha256, 
                'hdri_path': row.get('hdri_path'),
                'hdri_name': hdri_name,
                'rendered': True
            })
        else:
            filtered_metadata.append(row)
                
    metadata = pd.DataFrame(filtered_metadata)
    print(f'Processing {len(metadata)} object-HDRI combinations...')

    # Process objects
    func = partial(_render, output_dir=opt.output_dir, num_views=opt.num_views)
    rendered = foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)