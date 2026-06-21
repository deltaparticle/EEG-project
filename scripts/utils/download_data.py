import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import glob
import time
import urllib.request
import boto3
from botocore import UNSIGNED
from botocore.config import Config

bucket_name = "openneuro.org"
dataset_id = "ds003626"
target_dir = r"d:\Temple Project\ds003626"

# Clean up leftover temporary files
print("Cleaning up leftover temporary files...")
temp_pattern = os.path.join(target_dir, "sub-*", "ses-*", "eeg", "*.bdf.*")
temp_files = glob.glob(temp_pattern)
cleaned_count = 0
for temp_file in temp_files:
    if not temp_file.endswith(".bdf"):
        try:
            print(f"Removing temporary file: {os.path.basename(temp_file)}")
            os.remove(temp_file)
            cleaned_count += 1
        except Exception as e:
            print(f"Error removing {temp_file}: {e}")
print(f"Cleaned up {cleaned_count} temporary files.\n")

# Use S3 API to list objects
print(f"Connecting to S3 bucket '{bucket_name}' to scan keys for '{dataset_id}'...")
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=bucket_name, Prefix=dataset_id + "/")

files_to_download = []
subjects_to_download = ["sub-06", "sub-07", "sub-08", "sub-09", "sub-10"]

for page in pages:
    if "Contents" in page:
        for obj in page["Contents"]:
            key = obj["Key"]
            size = obj["Size"]
            relative_path = key[len(dataset_id)+1:] # strip 'ds003626/'
            if not relative_path:
                continue
                
            # Filter subjects and root metadata
            should_download = False
            if any(relative_path.startswith(sub + "/") for sub in subjects_to_download):
                should_download = True
            elif "/" not in relative_path:
                should_download = True
                
            if should_download:
                files_to_download.append((key, relative_path, size))

print(f"Scan complete. Found {len(files_to_download)} files to check.\n")

def download_file_http_resume(key, dest_path, expected_size):
    # Path-style URL avoids SSL certificate validation issues on buckets with dots
    url = f"https://s3.amazonaws.com/{bucket_name}/{key}"
    
    # Check local file size
    local_size = 0
    if os.path.exists(dest_path):
        local_size = os.path.getsize(dest_path)
        if local_size == expected_size:
            print(f" -> Skipping (already complete): {os.path.basename(dest_path)}")
            return True
        elif local_size > expected_size:
            print(f" -> Local size ({local_size}) is larger than expected ({expected_size}). Re-downloading from scratch.")
            try:
                os.remove(dest_path)
            except:
                pass
            local_size = 0

    print(f" -> Downloading {os.path.basename(dest_path)}")
    print(f"    Expected size: {expected_size / (1024*1024):.2f} MB")
    if local_size > 0:
        print(f"    Resuming from: {local_size / (1024*1024):.2f} MB")

    # Configure HTTP request
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
    if local_size > 0:
        req.add_header('Range', f'bytes={local_size}-')

    mode = 'ab' if local_size > 0 else 'wb'
    max_retries = 3
    
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, mode) as out_file:
                # Restart from scratch if server does not support Range
                status = response.getcode()
                if status != 206 and local_size > 0:
                    print("    Server did not support resume. Restarting from scratch...")
                    out_file.close()
                    try:
                        os.remove(dest_path)
                    except:
                        pass
                    return download_file_http_resume(key, dest_path, expected_size)
                
                downloaded = local_size
                chunk_size = 1024 * 1024 # 1 MB chunks
                last_print_time = time.time()
                
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    
                    # Print progress periodically
                    now = time.time()
                    if now - last_print_time > 3.0:
                        percent = (downloaded / expected_size) * 100
                        print(f"    Progress: {downloaded / (1024*1024):.2f} / {expected_size / (1024*1024):.2f} MB ({percent:.2f}%)")
                        last_print_time = now
                
                # Success
                percent = (downloaded / expected_size) * 100
                print(f"    Progress: {downloaded / (1024*1024):.2f} / {expected_size / (1024*1024):.2f} MB ({percent:.2f}%)")
                print(f" -> Finished downloading {os.path.basename(dest_path)}\n")
                return True
        except Exception as e:
            print(f"    Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                print("    Waiting 5 seconds before retrying...")
                time.sleep(5)
            else:
                print(f"    Failed to download after {max_retries} attempts.")
                return False

# Run downloads
download_count = 0
for idx, (key, relative_path, size) in enumerate(files_to_download, 1):
    dest_path = os.path.join(target_dir, relative_path.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    print(f"[{idx}/{len(files_to_download)}] Checking {relative_path}...")
    success = download_file_http_resume(key, dest_path, size)
    if success:
        download_count += 1

print(f"\nAll operations complete. Downloaded/Verified {download_count} files successfully.")
