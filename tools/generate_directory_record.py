"""
Generate a full directory record (CSV + JSON) for the project root.
Writes:
- DIRECTORY_RECORD_FULL.csv
- DIRECTORY_RECORD_FULL.json

Usage: python tools/generate_directory_record.py
"""
import os
import sys
import csv
import json
from datetime import datetime
from src.utils import paths

ROOT = str(paths.PROJECT_ROOT)
OUT_CSV = os.path.join(ROOT, 'DIRECTORY_RECORD_FULL.csv')
OUT_JSON = os.path.join(ROOT, 'DIRECTORY_RECORD_FULL.json')

records = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    rel_dir = os.path.relpath(dirpath, ROOT)
    if rel_dir == '.':
        rel_dir = ''
    # directories
    for d in sorted(dirnames):
        full = os.path.join(dirpath, d)
        try:
            st = os.stat(full)
            mtime = datetime.fromtimestamp(st.st_mtime).isoformat()
            size = st.st_size
        except Exception:
            mtime = ''
            size = ''
        records.append({
            'path': os.path.join(rel_dir, d).replace('\\', '/').lstrip('/'),
            'type': 'dir',
            'size_bytes': size,
            'mtime': mtime,
        })
    # files
    for f in sorted(filenames):
        full = os.path.join(dirpath, f)
        try:
            st = os.stat(full)
            mtime = datetime.fromtimestamp(st.st_mtime).isoformat()
            size = st.st_size
        except Exception:
            mtime = ''
            size = ''
        records.append({
            'path': os.path.join(rel_dir, f).replace('\\', '/').lstrip('/'),
            'type': 'file',
            'size_bytes': size,
            'mtime': mtime,
        })

# write CSV
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['path','type','size_bytes','mtime'])
    writer.writeheader()
    for r in records:
        writer.writerow(r)

# write JSON
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(records)} records to:\n - {OUT_CSV}\n - {OUT_JSON}")
