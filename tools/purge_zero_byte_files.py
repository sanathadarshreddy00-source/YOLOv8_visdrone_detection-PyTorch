import csv
import os
import shutil
from datetime import datetime, timezone
from src.utils import paths

DRY_RUN = True
ROOT = str(paths.PROJECT_ROOT)
CSV_IN = os.path.join(ROOT, 'DIRECTORY_RECORD_FULL.csv')

def purge_assets():
    if not os.path.exists(CSV_IN):
        print(f"❌ Cannot find {CSV_IN}")
        return

    print(f"--- Global Purge (Dry Run: {DRY_RUN}) ---")
    
    to_delete = []

    with open(CSV_IN, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Standardize keys
            data = {k.lower(): v for k, v in row.items()}
            path = data.get('path')
            item_type = str(data.get('type', '')).lower()
            
            try:
                size = int(data.get('size_bytes', -1))
            except:
                size = -1

            full_path = os.path.join(ROOT, path)
            if not os.path.exists(full_path):
                continue

            # CASE 1: Zero-byte File
            if item_type == 'file' and size == 0:
                if not path.endswith('.py'): # Safety: keep python files
                    to_delete.append(('file', full_path, path))

            # CASE 2: Empty Directory
            elif item_type == 'directory':
                # Check if folder is actually empty on disk
                if os.path.isdir(full_path) and not os.listdir(full_path):
                    to_delete.append(('folder', full_path, path))

    # Execution phase
    for category, full, rel in to_delete:
        if DRY_RUN:
            print(f"[DRY RUN] Would delete {category}: {rel}")
        else:
            try:
                if category == 'file':
                    os.remove(full)
                else:
                    os.rmdir(full) # Only removes if empty
                print(f"✅ Deleted {category}: {rel}")
            except Exception as e:
                print(f"❌ Failed to delete {rel}: {e}")

    print(f"\nTotal items identified: {len(to_delete)}")
    
    if not DRY_RUN and len(to_delete) > 0:
        print("\nRegenerating directory record...")
        os.system(f"python tools{os.sep}generate_directory_record.py")

if __name__ == "__main__":
    purge_assets()