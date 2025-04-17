import os
import json
import copy
import glob
import sys
import importlib
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
    
    # load metadata
    metadata = metadata.to_dict('records')

    # processing objects
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
                    record = func(file, sha256)
                    if record is not None:
                        if isinstance(record, dict):
                            records.append(record)
                        elif isinstance(record, list):
                            records.extend(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    return pd.DataFrame.from_records(records)


def _render(file_path, sha256, output_dir, num_views, hdri_paths):
    base_folder = os.path.join(output_dir, 'renders', sha256)
    records = []
    
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
    
    if hdri_paths:
        for hdri_path in hdri_paths:
            hdri_name = os.path.splitext(os.path.basename(hdri_path))[0]
            output_folder = os.path.join(base_folder, hdri_name)
            args_run = args + ['--hdri_path', hdri_path,
                               '--output_folder', output_folder]
            call(args_run, stdout=DEVNULL, stderr=DEVNULL)
            if os.path.exists(os.path.join(output_folder, 'transforms.json')):
                records.append({
                    'sha256': sha256,
                    'hdri':   hdri_name,
                    'rendered': True
                })
    else:
        args += ['--output_folder', base_folder]
        call(args, stdout=DEVNULL, stderr=DEVNULL)
    
        if os.path.exists(os.path.join(base_folder, 'transforms.json')):
            records.append({
                'sha256': sha256,
                'hdri':   None,
                'rendered': True
            })        

    return records


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
    # new HDRI args
    parser.add_argument('--hdri_list', type=str,
                        help='Comma-separated list of HDRI .exr file paths')
    parser.add_argument('--hdri_dir', type=str,
                        help='Directory containing HDRI .exr files')
    parser.add_argument('--hdri_strength', type=float, default=1.0,
                        help='Strength multiplier for HDRI environment lighting')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)
    
    # Find all hdri files
    hdri_paths = []
    if opt.hdri_dir:
        hdri_paths = sorted(glob.glob(os.path.join(opt.hdri_dir, '*.exr')))    
    elif opt.hdri_list:
        hdri_paths = [p.strip() for p in opt.hdri_list.split(',') if p.strip()]

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
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

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_render, output_dir=opt.output_dir, num_views=opt.num_views, hdri_paths=hdri_paths)
    rendered = foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)
