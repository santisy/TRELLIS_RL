import os
import argparse
import pandas as pd
import glob
import hashlib
import tqdm

def compute_sha256(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def create_metadata(objects_dir, hdri_dir=None, output_dir=None):
    records = []
    
    # Find all 3D objects (obj, fbx, glb, blend)
    object_files = []
    for ext in ['*.obj', '*.fbx', '*.glb', '*.blend']:
        object_files.extend(glob.glob(os.path.join(objects_dir, '**', ext), recursive=True))
    
    # Find all HDRI files if provided
    hdri_files = []
    if hdri_dir:
        hdri_files = [os.path.realpath(p) for p in glob.glob(os.path.join(hdri_dir, '*.exr'))]
    
    # Create metadata for each object
    for obj_path in tqdm.tqdm(object_files, desc="Processing objects"):
        real_path = os.path.realpath(obj_path)
        sha256 = compute_sha256(obj_path)
        
        if hdri_files:
            # Create N×M entries (one for each object-HDRI combination)
            for hdri_path in hdri_files:
                hdri_name = os.path.splitext(os.path.basename(hdri_path))[0]
                records.append({
                    'local_path': real_path,
                    'sha256': sha256,
                    'hdri_path': hdri_path,
                    'hdri_name': hdri_name,
                    'rendered': False,
                    'aesthetic_score': 1.0  # Default value
                })
        else:
            # Just object entries without HDRI
            records.append({
                'local_path': real_path,
                'sha256': sha256,
                'hdri_path': None,
                'hdri_name': None,
                'rendered': False,
                'aesthetic_score': 1.0  # Default value
            })
    
    # Create DataFrame and save to CSV
    metadata = pd.DataFrame.from_records(records)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, 'metadata.csv')
        metadata.to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")
        
    return metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create metadata.csv for rendering objects with environment maps')
    parser.add_argument('--objects_dir', type=str, required=True, 
                        help='Directory containing 3D object files')
    parser.add_argument('--hdri_dir', type=str, default=None,
                        help='Directory containing HDRI .exr files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the metadata.csv file')
    
    args = parser.parse_args()
    create_metadata(args.objects_dir, args.hdri_dir, args.output_dir)