#!/usr/bin/env python3
"""
Helper script to get Open Images image IDs.
This script provides a working set of image IDs for quick setup.
"""

import sys
import os

# Note: We don't use hardcoded sample IDs since they may not exist.
# Instead, we rely on S3 listing to get real image IDs.

def get_image_ids_from_s3(split, max_images=5000):
    """
    Try to get image IDs by listing S3 bucket objects.
    This is slower but works without metadata files.
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        bucket_name = 'open-images-dataset'
        prefix = f'{split}/'
        
        image_ids = []
        paginator = s3.get_paginator('list_objects_v2')
        
        print(f"Fetching image IDs from S3 bucket for {split} split...", file=sys.stderr)
        print("This may take a few moments...", file=sys.stderr)
        
        page_count = 0
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, MaxKeys=1000):
            page_count += 1
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Extract image ID from key like "train/abc123.jpg"
                    if key.endswith('.jpg'):
                        image_id = key.split('/')[-1].replace('.jpg', '')
                        if image_id and len(image_id) > 0:
                            image_ids.append(f"{split}/{image_id}")
                            if max_images > 0 and len(image_ids) >= max_images:
                                print(f"Found {len(image_ids)} images (stopping at requested limit)", file=sys.stderr)
                                return image_ids
            
            # Progress update
            if page_count % 10 == 0:
                print(f"Processed {page_count} pages, found {len(image_ids)} images so far...", file=sys.stderr)
        
        print(f"Found {len(image_ids)} total images", file=sys.stderr)
        return image_ids
    except ImportError:
        print(f"Warning: boto3 not installed. Install with: pip install boto3", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Could not fetch from S3: {e}", file=sys.stderr)
        return None

def generate_image_list(split='train', max_images=5000, use_s3=True):
    """
    Generate a list of image IDs for the specified split.
    
    Args:
        split: Dataset split ('train', 'validation', or 'test')
        max_images: Maximum number of images to return (0 means all available)
        use_s3: Whether to try fetching from S3 first
    """
    image_ids = []
    
    # Try to get from S3 if requested
    if use_s3:
        # If max_images is 0, we want all images, so pass a very large number
        s3_max = max_images if max_images > 0 else 999999999
        s3_ids = get_image_ids_from_s3(split, s3_max)
        if s3_ids:
            # If max_images is 0, return all; otherwise limit
            if max_images == 0:
                return s3_ids
            else:
                return s3_ids[:max_images]
    
    # If S3 failed and we have no IDs, try to install boto3 and retry
    if not image_ids:
        print("Attempting to install boto3...", file=sys.stderr)
        import subprocess
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'boto3'])
            print("boto3 installed. Retrying S3 fetch...", file=sys.stderr)
            s3_ids = get_image_ids_from_s3(split, max_images)
            if s3_ids:
                return s3_ids[:max_images] if max_images > 0 else s3_ids
        except:
            pass
    
    # If still no IDs, return empty list (user will need to provide their own list)
    if not image_ids:
        print("Error: Could not fetch image IDs automatically.", file=sys.stderr)
        print("Please install boto3: pip install boto3", file=sys.stderr)
        print("Or provide an image list file using --image-list option.", file=sys.stderr)
    
    return image_ids

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: get_open_images_ids.py <split> <output_file> [max_images] [--no-s3]", file=sys.stderr)
        sys.exit(1)
    
    split = sys.argv[1]
    output_file = sys.argv[2]
    max_images = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 5000
    use_s3 = '--no-s3' not in sys.argv
    
    image_ids = generate_image_list(split, max_images, use_s3)
    
    with open(output_file, 'w') as f:
        for img_id in image_ids:
            f.write(f"{img_id}\n")
    
    print(f"Generated {len(image_ids)} image IDs for {split} split", file=sys.stderr)

