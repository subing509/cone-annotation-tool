"""Read-only audit of originals and reproducible blocked training split."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import shutil
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'cone_project' / 'data'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / 'JPEGImage').glob('*.jpg'), key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**12, p.stem))
    rows, excluded, hashes = [], [], {}
    sizes, classes = Counter(), Counter()
    thumbnails = []
    for idx, p in enumerate(files):
        label = ROOT / 'label' / (p.stem + '.txt')
        if not label.exists():
            excluded.append({'image': p.name, 'reason': 'missing_label'})
            continue
        try:
            with Image.open(p) as im:
                im.load()
                width, height = im.size
                if idx % 450 == 0:
                    thumbs = ImageOps.contain(im.convert('RGB'), (480, 220))
                    canvas = Image.new('RGB', (480, 250), 'white')
                    canvas.paste(thumbs, (0, 25))
                    ImageDraw.Draw(canvas).text((8, 5), f'{p.name} {width}x{height}', fill='black')
                    thumbnails.append(canvas)
            lines = label.read_text(encoding='utf-8-sig').splitlines()
            if not lines:
                excluded.append({'image': p.name, 'reason': 'empty_label_unreviewed'})
                continue
            counts = Counter()
            for line in lines:
                a = [float(x) for x in line.split()]
                assert len(a) == 5 and all(math.isfinite(v) for v in a)
                cls, x, y, w, h = a
                assert cls in (0, 1, 2) and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                assert x-w/2 >= -1e-6 and x+w/2 <= 1+1e-6 and y-h/2 >= -1e-6 and y+h/2 <= 1+1e-6
                counts[int(cls)] += 1
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in hashes:
                excluded.append({'image': p.name, 'reason': 'exact_duplicate', 'duplicate_of': hashes[digest]})
                continue
            hashes[digest] = p.name
            rows.append({'image': p.name, 'label': label.name, 'width': width, 'height': height, 'sha256': digest, 'classes': dict(counts)})
            sizes[f'{width}x{height}'] += 1
            classes.update(counts)
        except Exception as e:
            excluded.append({'image': p.name, 'reason': 'invalid_image_or_label', 'detail': str(e)})
        if idx % 1000 == 0:
            print(f'audited {idx}/{len(files)}', flush=True)
    # Unknown source: ordered, contiguous 70/15/15 split with boundary guards.
    n = len(rows)
    b1, b2 = int(n*.70), int(n*.85)
    guard = 25
    splits = {'train': rows[:b1-guard], 'val': rows[b1+guard:b2-guard], 'test': rows[b2+guard:]}
    for row in rows[b1-guard:b1+guard] + rows[b2-guard:b2+guard]:
        excluded.append({'image': row['image'], 'reason': 'split_boundary_guard'})
    manifest = []
    for split, items in splits.items():
        dest_i = OUT / 'images' / split
        dest_l = OUT / 'labels' / split
        dest_i.mkdir(parents=True, exist_ok=True)
        dest_l.mkdir(parents=True, exist_ok=True)
        for row in items:
            shutil.copy2(ROOT / 'JPEGImage' / row['image'], dest_i / row['image'])
            shutil.copy2(ROOT / 'label' / row['label'], dest_l / row['label'])
            manifest.append(dict(row, split=split))
    (OUT / 'data.yaml').write_text(f'path: "{OUT.as_posix()}"\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: red\n  1: blue\n  2: yellow\n', encoding='utf-8')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    split_counts = {s: dict(Counter({str(k): sum(r['classes'].get(k, 0) for r in items) for k in (0,1,2)})) for s,items in splits.items()}
    report = {'original_images': len(files), 'valid_unique_nonempty': n, 'split_method': 'numeric ordered contiguous 70/15/15, 25 image guards on each side of boundaries', 'seed': 42, 'source_unknown': True, 'limitation': 'Near duplicates and unknown shared source across splits remain possible; independent new-scene test required for deployment acceptance.', 'sizes': dict(sizes), 'class_counts': dict(classes), 'split_images': {s:len(v) for s,v in splits.items()}, 'split_class_counts': split_counts, 'excluded': excluded}
    (OUT / 'audit_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if thumbnails:
        sheet = Image.new('RGB', (1440, 250*((len(thumbnails)+2)//3)), '#ddd')
        for k,thumb in enumerate(thumbnails): sheet.paste(thumb, ((k%3)*480,(k//3)*250))
        sheet.save(OUT / 'dataset_samples.jpg')
    print(json.dumps({k:v for k,v in report.items() if k != 'excluded'}, ensure_ascii=False, indent=2), flush=True)
    print('excluded:', len(excluded), flush=True)


if __name__ == '__main__':
    main()
