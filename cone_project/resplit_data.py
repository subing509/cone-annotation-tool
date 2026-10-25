"""Balance whole ordered groups; never randomly split individual frames."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import random
import shutil
from PIL import Image

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
OUT = BASE / 'data'


def main():
    report = json.loads((OUT / 'audit_report.json').read_text(encoding='utf-8'))
    rows = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    for x in report['excluded']:
        if x['reason'] != 'split_boundary_guard': continue
        p = ROOT / 'JPEGImage' / x['image']
        label = ROOT / 'label' / (p.stem + '.txt')
        with Image.open(p) as im: w,h=im.size
        c=Counter(int(line.split()[0]) for line in label.read_text().splitlines())
        rows.append(dict(image=p.name,label=label.name,width=w,height=h,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),classes=dict(c)))
    rows.sort(key=lambda r:int(Path(r['image']).stem))
    groups=[rows[k:k+250] for k in range(0,len(rows),250)]
    stats=[]
    for group in groups:
        stats.append([len(group)] + [sum(r['classes'].get(str(k),r['classes'].get(k,0)) for r in group) for k in range(3)])
    totals=[sum(s[k] for s in stats) for k in range(4)]
    rng=random.Random(42)
    best=None
    for _ in range(6000):
        order=list(range(len(groups))); rng.shuffle(order)
        n1=round(len(groups)*.7); n2=round(len(groups)*.15)
        assignments={'train':order[:n1], 'val':order[n1:n1+n2], 'test':order[n1+n2:]}
        score=0
        for split,ids in assignments.items():
            target={'train':.7,'val':.15,'test':.15}[split]
            for k in range(4):
                fraction=sum(stats[j][k] for j in ids)/totals[k]
                score+=((fraction-target)/target)**2
        if best is None or score<best[0]: best=(score,assignments)
    assignments=best[1]
    membership={j:s for s,ids in assignments.items() for j in ids}
    manifest=[]; excluded=[r for r in report['excluded'] if r['reason']!='split_boundary_guard']
    for j,group in enumerate(groups):
        split=membership[j]
        lo=10 if j>0 and membership[j-1]!=split else 0
        hi=len(group)-10 if j+1<len(groups) and membership[j+1]!=split else len(group)
        for r in group[:lo]+group[hi:]: excluded.append(dict(image=r['image'],reason='split_boundary_guard'))
        for r in group[lo:hi]: manifest.append(dict(r,split=split,group_id=j))
    # Only generated copies within data/ are moved; originals remain untouched.
    wanted={r['image']:r['split'] for r in manifest}
    for split in ('train','val','test'):
        for folder in ('images','labels'): (OUT/folder/split).mkdir(parents=True,exist_ok=True)
        for p in list((OUT/'images'/split).glob('*.jpg')):
            if wanted.get(p.name)==split: continue
            target=OUT/'images'/wanted.get(p.name,'guard')/p.name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(p),str(target))
            lp=OUT/'labels'/split/(p.stem+'.txt')
            lt=OUT/'labels'/wanted.get(p.name,'guard')/lp.name
            lt.parent.mkdir(parents=True,exist_ok=True)
            if lp.exists(): shutil.move(str(lp),str(lt))
    for r in manifest:
        ip=OUT/'images'/r['split']/r['image']; lp=OUT/'labels'/r['split']/r['label']
        if not ip.exists(): shutil.copy2(ROOT/'JPEGImage'/r['image'],ip)
        if not lp.exists(): shutil.copy2(ROOT/'label'/r['label'],lp)
    report.update(split_method='250 ordered valid-image groups, seeded whole-group class-balanced allocation; 10-image guards on either side of cross-split boundaries', group_seed=42,
                  split_images=dict(Counter(r['split'] for r in manifest)),
                  split_class_counts={s:{str(k):sum(r['classes'].get(str(k),r['classes'].get(k,0)) for r in manifest if r['split']==s) for k in range(3)} for s in ('train','val','test')},excluded=excluded)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('split_method','split_images','split_class_counts')},indent=2))


if __name__=='__main__': main()
