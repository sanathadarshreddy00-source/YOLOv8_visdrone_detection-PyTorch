"""Create a small sanitized inventory summary suitable for committing.

Writes:
- DIRECTORY_RECORD_SUMMARY.json
- DIRECTORY_RECORD_SUMMARY.csv

Filters to the requested paths: `configs/`, `src/`, `scripts/`, `tools/`,
`README.md`, `DEVELOPER_GUIDE.md`, and top-level `MIGRATION_MANIFEST_*.csv` plus
`runs/detect/` entries but limits the runs listing to recent 20 items.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_JSON = ROOT / 'DIRECTORY_RECORD_FULL.json'
OUT_JSON = ROOT / 'DIRECTORY_RECORD_SUMMARY.json'
OUT_CSV = ROOT / 'DIRECTORY_RECORD_SUMMARY.csv'

if not IN_JSON.exists():
    raise SystemExit(f"Missing {IN_JSON}, run tools/generate_directory_record.py first")

with open(IN_JSON, 'r', encoding='utf-8') as f:
    records = json.load(f)

keep_prefixes = (
    'configs/', 'src/', 'scripts/', 'tools/',
    'README.md', 'DEVELOPER_GUIDE.md', 'MIGRATION_MANIFEST_', 'MIGRATION_MANIFEST',
)

summary = []
runs_entries = [r for r in records if r['path'].startswith('runs/detect/')]
# keep only latest 200 runs entries (by mtime if present)
def mtime_key(r):
    return r.get('mtime') or ''
runs_entries_sorted = sorted(runs_entries, key=mtime_key, reverse=True)[:200]

for r in records:
    p = r['path']
    if any(p == kp or p.startswith(kp) for kp in keep_prefixes):
        summary.append(r)

# add the filtered runs entries (recent subset)
for r in runs_entries_sorted:
    if r not in summary:
        summary.append(r)

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

with open(OUT_CSV, 'w', encoding='utf-8') as f:
    f.write('path,type,size_bytes,mtime\n')
    for r in summary:
        line = f"{r.get('path','')},{r.get('type','')},{r.get('size_bytes','')},{r.get('mtime','')}\n"
        f.write(line)

print(f"Wrote summary: {OUT_JSON} ({len(summary)} entries)")
