#!/usr/bin/env python3

import os
import requests
from urllib.parse import urlparse
import shutil
from pathlib import Path
import time
from tqdm import tqdm

# Array of downloads: each item contains [filename_to_save_as, download_url]
DOWNLOADS = [
    ["v1.0-mini.tgz", "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz"],
    # ["model_weights.pth", "https://example.com/model_weights.pth"],
    # ["sample_images.tar.gz", "https://example.com/sample_images.tar.gz"],
    # Add more downloads as needed
]

def download_file(url, destination_path):
    """
    Download a file from the given URL to the specified destination path
    with progress bar.
    """
    try:
        # Create a session to reuse connection
        session = requests.Session()
        
        # Make a HEAD request first to get the file size
        response = session.head(url, allow_redirects=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        
        # Stream the download with progress bar
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise exception for bad responses
        
        # Setup progress bar
        progress_bar = tqdm(
            total=total_size_in_bytes, 
            unit='iB', 
            unit_scale=True,
            desc=f"Downloading {os.path.basename(destination_path)}"
        )
        
        # Write to file with progress updates
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    progress_bar.update(len(chunk))
                    file.write(chunk)
        
        progress_bar.close()
        
        # Check if download was complete
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print(f"ERROR: Downloaded file size does not match expected size for {url}")
            return False
            
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        # Remove partially downloaded file if it exists
        if os.path.exists(destination_path):
            os.remove(destination_path)
        return False

def download_all():
    """
    Download all files specified in the DOWNLOADS array.
    Each file is placed in its own subdirectory within the downloads directory.
    Skips downloads if the file already exists.
    """
    # Create downloads directory if it doesn't exist
    downloads_dir = Path("downloads")
    downloads_dir.mkdir(exist_ok=True)
    
    success_count = 0
    failure_count = 0
    skipped_count = 0
    
    for filename, url in DOWNLOADS:
        # Create a unique directory name based on the filename
        dir_name = os.path.splitext(filename)[0]
        download_dir = downloads_dir / dir_name
        
        # Full path for the downloaded file
        file_path = download_dir / filename
        
        # Check if file already exists
        if file_path.exists():
            print(f"\nSkipping {filename} - File already exists at {file_path}")
            skipped_count += 1
            continue
        
        # Create directory if it doesn't exist
        download_dir.mkdir(exist_ok=True)
        
        print(f"\nDownloading {filename} to {file_path}")
        
        # Download the file
        start_time = time.time()
        success = download_file(url, file_path)
        end_time = time.time()
        
        # Update counters and print results
        if success:
            success_count += 1
            print(f"Successfully downloaded {filename} in {end_time - start_time:.2f} seconds")
        else:
            failure_count += 1
            print(f"Failed to download {filename}")
    
    # Print summary
    print(f"\nDownload Summary:")
    print(f"- Successfully downloaded: {success_count} files")
    print(f"- Failed downloads: {failure_count} files")
    print(f"- Skipped downloads (already exist): {skipped_count} files")
    print(f"- Total files processed: {len(DOWNLOADS)} files")
    
    return success_count, failure_count, skipped_count

if __name__ == "__main__":
    print("Starting download process...")
    download_all()
    print("Download process completed.") 