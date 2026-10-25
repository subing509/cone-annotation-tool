"""Local Windows annotation tool. Originals are never modified by this service."""
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs
import argparse
import base64
import binascii
import csv
import hashlib
import json
import mimetypes
import os
import re
import shutil
import socket
import threading
import time
import traceback
import uuid

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
for name in ('.ultralytics/Ultralytics','.matplotlib','tool_data/reviews','tool_data/jobs','tool_data/exports'):
    (BASE/name).mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(BASE/'.ultralytics')
os.environ['MPLCONFIGDIR']=str(BASE/'.matplotlib')
os.environ['YOLO_AUTOINSTALL']='false'
IMAGE_EXT={'.jpg','.jpeg','.png','.bmp','.webp'}
VIDEO_EXT={'.mp4','.avi','.mov','.mkv'}
POOL=ThreadPoolExecutor(max_workers=1)
LOCK=threading.RLock()
MODEL_LOCK=threading.Lock()
MODEL=None
JOBS={}
IMPORTS={}
NAMES=['red','blue','yellow']


def safe_path(value, *, file=False):
    p=(ROOT/str(value)).resolve()
    if not p.is_relative_to(ROOT.resolve()): raise ValueError('Path must be inside dataset workspace')
    if file and not p.is_file(): raise ValueError('File does not exist')
    return p


def file_id(p): return hashlib.sha256(str(p.relative_to(ROOT)).encode()).hexdigest()[:20]


def atomic_json(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def validate_boxes(boxes,width,height):
    import math
    result=[]
    if not isinstance(boxes,list) or len(boxes)>2000: raise ValueError('Invalid boxes')
    for b in boxes:
        c=b.get('class_id'); xy=b.get('xyxy')
        if type(c) is not int or c not in (0,1,2) or not isinstance(xy,list) or len(xy)!=4:
            raise ValueError('Class must be 0/red, 1/blue, 2/yellow')
        xy=list(map(float,xy))
        if not all(math.isfinite(v) for v in xy): raise ValueError('Nonfinite coordinate')
        x1,y1,x2,y2=xy
        if not (0<=x1<x2<=width and 0<=y1<y2<=height): raise ValueError('Box outside image or empty')
        result.append(dict(class_id=c,xyxy=xy))
    return result


def image_info(p):
    from PIL import Image
    with Image.open(p) as im: width,height=im.size
    review=BASE/'tool_data/reviews'/f'{file_id(p)}.json'
    if review.exists():
        data=json.loads(review.read_text(encoding='utf-8'))
        return dict(path=p.relative_to(ROOT).as_posix(),width=width,height=height,boxes=data['boxes'],reviewed=True)
    label=ROOT/'label'/(p.stem+'.txt') if p.parent==ROOT/'JPEGImage' else p.parent.parent/'labels'/p.parent.name/(p.stem+'.txt')
    boxes=[]
    if label.exists():
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            try:
                c,x,y,w,h=map(float,line.split())
                boxes.append(dict(class_id=int(c),xyxy=[max(0,(x-w/2)*width),max(0,(y-h/2)*height),min(width,(x+w/2)*width),min(height,(y+h/2)*height)]))
            except ValueError: pass
    return dict(path=p.relative_to(ROOT).as_posix(),width=width,height=height,boxes=boxes,reviewed=False)


def confidence_settings(conf):
    import math
    if isinstance(conf,dict):
        if set(conf)!=set(NAMES): raise ValueError('Provide red, blue and yellow confidence thresholds')
        thresholds={name:float(conf[name]) for name in NAMES}
    else:
        thresholds={name:float(conf) for name in NAMES}
    if not all(math.isfinite(v) and .01<=v<=.99 for v in thresholds.values()):
        raise ValueError('Each confidence threshold must be between 0.01 and 0.99')
    return thresholds


def position_settings(value=None):
    import math
    value={} if value is None else value
    if not isinstance(value,dict): raise ValueError('Invalid position settings')
    height=float(value.get('cone_height_m',.45)); fov=float(value.get('hfov_deg',90))
    enabled=value.get('enabled',True)
    if type(enabled) is not bool or not math.isfinite(height) or not .1<=height<=2 or not math.isfinite(fov) or not 20<=fov<=160:
        raise ValueError('Cone height must be 0.1–2 m; horizontal FOV 20–160 degrees')
    return dict(enabled=enabled,cone_height_m=height,hfov_deg=fov)


def estimate_position(box,width,height,settings):
    import math
    x1,y1,x2,y2=box['xyxy']
    result=dict(dist_m=None,x_m=None,z_m=None,position_valid=False,dist_source='size_approx')
    # Full visible height is required; image-edge truncation makes this estimate unreliable.
    if not settings['enabled'] or y2-y1<6 or min(x1,y1)<2 or x2>width-2 or y2>height-2: return result
    focal=width/(2*math.tan(math.radians(settings['hfov_deg'])/2))
    z=focal*settings['cone_height_m']/(y2-y1)
    x=((x1+x2)/2-width/2)*z/focal
    result.update(dist_m=round(z,2),x_m=round(x,2),z_m=round(z,2),position_valid=True)
    return result


def annotate(image,boxes,settings):
    import cv2
    result=image.copy(); h,w=result.shape[:2]
    colors=[(91,85,255),(255,150,66),(52,197,245)]
    for b in boxes:
        x1,y1,x2,y2=map(round,b['xyxy']); color=colors[b['class_id']]
        cv2.rectangle(result,(x1,y1),(x2,y2),color,2)
        lines=[f"{NAMES[b['class_id']]}_cone {b['score']:.2f}"]
        if settings['enabled']:
            lines.append(f"~Z:{b['z_m']:.1f}m X:{b['x_m']:+.1f}m" if b['position_valid'] else 'Position: N/A')
        top=max(20,y1-18*len(lines))
        for i,line in enumerate(lines):
            y=min(h-5,top+18*i); tw=cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,.48,1)[0][0]
            left=max(0,min(x1,w-tw-8))
            cv2.rectangle(result,(left,max(0,y-15)),(min(w,left+tw+8),y+4),color,-1)
            cv2.putText(result,line,(left+3,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1,cv2.LINE_AA)
    if settings['enabled']:
        cv2.putText(result,'ROUGH ESTIMATE | X: right (+) / left (-), Z: forward',(8,h-12),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    return result


def infer(source,conf=.25,imgsz=640,position=None):
    global MODEL
    thresholds=confidence_settings(conf)
    position=position_settings(position)
    if int(imgsz) not in (640,960,1280): raise ValueError('Invalid inference settings')
    with MODEL_LOCK:
        if MODEL is None:
            from ultralytics import YOLO
            weights=BASE/'models/best.pt'
            if not weights.is_file(): raise ValueError('Configure cone_project/models/best.pt first; see README.md')
            MODEL=YOLO(str(weights))
            assert list(MODEL.names.values()) in (['red','blue','yellow'],['red_cone','blue_cone','yellow_cone'])
        import torch
        prediction=MODEL.predict(source=source,conf=min(thresholds.values()),imgsz=int(imgsz),device=0 if torch.cuda.is_available() else 'cpu',verbose=False)[0]
        # Filter before rendering so UI, video, CSV, JSONL and YOLO labels agree.
        limits=torch.tensor([thresholds[name] for name in NAMES],device=prediction.boxes.conf.device)
        keep=prediction.boxes.conf>=limits[prediction.boxes.cls.long()]
        prediction.boxes=prediction.boxes[keep]
        boxes=[dict(class_id=int(c),xyxy=[float(v) for v in xy],score=float(s)) for xy,c,s in zip(prediction.boxes.xyxy.cpu().tolist(),prediction.boxes.cls.cpu().tolist(),prediction.boxes.conf.cpu().tolist())]
        h,w=prediction.orig_img.shape[:2]
        for box in boxes: box.update(estimate_position(box,w,h,position))
        annotated=annotate(prediction.orig_img,boxes,position)
    return boxes,annotated


def yolo_text(boxes,w,h):
    return ''.join(f"{b['class_id']} {(b['xyxy'][0]+b['xyxy'][2])/2/w:.8f} {(b['xyxy'][1]+b['xyxy'][3])/2/h:.8f} {(b['xyxy'][2]-b['xyxy'][0])/w:.8f} {(b['xyxy'][3]-b['xyxy'][1])/h:.8f}\n" for b in boxes)


def job_update(job,**kw):
    with LOCK:
        job.update(kw)
        atomic_json(BASE/'tool_data/jobs'/f"{job['id']}.json",job)


def run_job(job,paths,conf,imgsz,position=None):
    position=position_settings(position)
    import cv2
    output=BASE/'tool_data/exports'/job['id']
    output.mkdir(parents=True)
    for folder in ('images','labels_yolo','annotated'): (output/folder).mkdir()
    job_update(job,state='running',output=output.relative_to(ROOT).as_posix())
    fields=['source','frame_idx','timestamp_ms','class_id','class','score','x1','y1','x2','y2','reviewed','dist_m','x_m','z_m','position_valid','dist_source']
    try:
        with (output/'detections.csv').open('w',encoding='utf-8-sig',newline='') as csvfile,(output/'detections.jsonl').open('w',encoding='utf-8') as jsonfile:
            writer=csv.DictWriter(csvfile,fieldnames=fields); writer.writeheader()
            for i,relative in enumerate(paths):
                if job.get('cancel_requested'): break
                p=safe_path(relative,file=True)
                try:
                    if p.suffix.lower() in IMAGE_EXT:
                        import shutil
                        from PIL import Image
                        with Image.open(p) as im: w,h=im.size
                        boxes,annotated=infer(str(p),conf,imgsz,position)
                        stem=f'{p.stem}_{file_id(p)}'
                        shutil.copy2(p,output/'images'/(stem+p.suffix))
                        (output/'labels_yolo'/f'{stem}.txt').write_text(yolo_text(boxes,w,h),encoding='utf-8')
                        cv2.imencode('.jpg',annotated)[1].tofile(str(output/'annotated'/f'{stem}.jpg'))
                        records=[(0,0,boxes)]
                        for frame_idx,timestamp,bs in records: write_frame(writer,jsonfile,relative,frame_idx,timestamp,bs)
                    elif p.suffix.lower() in VIDEO_EXT:
                        cap=cv2.VideoCapture(str(p)); video_writer=None
                        try:
                            if not cap.isOpened(): raise ValueError('Video cannot be decoded')
                            fps=cap.get(cv2.CAP_PROP_FPS) or 30; index=0
                            while not job.get('cancel_requested'):
                                ok,frame=cap.read()
                                if not ok: break
                                h,w=frame.shape[:2]
                                if video_writer is None:
                                    video_writer=cv2.VideoWriter(str(output/'annotated'/f'{p.stem}_{file_id(p)}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                                    if not video_writer.isOpened(): raise ValueError('Video encoder unavailable')
                                boxes,annotated=infer(frame,conf,imgsz,position)
                                video_writer.write(annotated)
                                write_frame(writer,jsonfile,relative,index,index*1000/fps,boxes)
                                index+=1
                                if index%20==0: job_update(job,frame=index)
                            if index==0: raise ValueError('Video contains no readable frames')
                        finally:
                            cap.release()
                            if video_writer is not None: video_writer.release()
                    else: raise ValueError('Unsupported media')
                except Exception as exc:
                    job['errors'].append(dict(path=relative,error=str(exc)))
                job_update(job,done=i+1)
        thresholds=confidence_settings(conf)
        atomic_json(output/'meta.json',dict(model=str(BASE/'models/best.pt'),model_sha256=hashlib.sha256((BASE/'models/best.pt').read_bytes()).hexdigest(),conf=min(thresholds.values()),conf_by_class=thresholds,imgsz=imgsz,classes=NAMES,reviewed=False,dist_m=None,distance_state='approximate_uncalibrated' if position['enabled'] else 'disabled',position=position,position_axes='camera X right-positive, Z optical-axis forward; metres',position_assumptions='Full cone height; square pixels; centered camera; no pitch or occlusion correction',video_codec='mp4v',errors=job['errors']))
        job_update(job,state='cancelled' if job.get('cancel_requested') else ('completed_with_errors' if job['errors'] else 'completed'))
    except Exception as exc: job_update(job,state='failed',error=str(exc))


def write_frame(writer,jsonfile,source,index,timestamp,boxes):
    jsonfile.write(json.dumps(dict(source=source,frame_idx=index,timestamp_ms=timestamp,boxes=boxes,reviewed=False,dist_m=None),ensure_ascii=False)+'\n')
    for b in boxes:
        writer.writerow(dict(source=source,frame_idx=index,timestamp_ms=timestamp,class_id=b['class_id'],**{'class':NAMES[b['class_id']]},score=b['score'],**dict(zip(['x1','y1','x2','y2'],b['xyxy'])),reviewed=0,**{key:b.get(key) for key in ('dist_m','x_m','z_m','position_valid','dist_source')}))


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def send_json(self,data,status=200):
        raw=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        try:
            parsed=urlparse(self.path); q=parse_qs(parsed.query)
            if parsed.path=='/':
                raw=(BASE/'web/index.html').read_bytes(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.end_headers(); self.wfile.write(raw)
            elif parsed.path=='/api/model':
                metadata=BASE/'active_model.json'
                active=json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else dict(model='YOLO11 cone detector',configured=(BASE/'models/best.pt').exists())
                self.send_json({k:v for k,v in active.items() if k!='source_file_hashes'})
            elif parsed.path=='/api/files':
                folder=safe_path(q.get('folder',['JPEGImage'])[0]); offset=max(0,int(q.get('offset',['0'])[0]))
                if not folder.is_dir(): raise ValueError('Folder does not exist')
                files=sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT|VIDEO_EXT),key=lambda p:(0,int(p.stem)) if p.stem.isdigit() else (1,p.name))
                self.send_json(dict(total=len(files),files=[dict(path=p.relative_to(ROOT).as_posix(),name=p.name,kind='image' if p.suffix.lower() in IMAGE_EXT else 'video',reviewed=(BASE/'tool_data/reviews'/f'{file_id(p)}.json').exists()) for p in files[offset:offset+60]]))
            elif parsed.path=='/api/image': self.send_json(image_info(safe_path(q['path'][0],file=True)))
            elif parsed.path=='/media':
                p=safe_path(q['path'][0],file=True)
                if p.suffix.lower() not in IMAGE_EXT|VIDEO_EXT: raise ValueError('Not media')
                self.send_response(200); self.send_header('Content-Type',mimetypes.guess_type(p.name)[0] or 'application/octet-stream'); self.send_header('Content-Length',str(p.stat().st_size)); self.end_headers()
                with p.open('rb') as f:
                    while chunk:=f.read(1024*1024): self.wfile.write(chunk)
            elif parsed.path=='/api/jobs':
                with LOCK: self.send_json(list(JOBS.values()))
            else: self.send_json(dict(error='Not found'),404)
        except (ValueError,KeyError,FileNotFoundError) as exc: self.send_json(dict(error=str(exc)),400)
        except Exception as exc: self.send_json(dict(error=str(exc)),500)
    def do_POST(self):
        try:
            # Do not allow cross-origin writes to a localhost service.
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
                return self.send_json(dict(error='Cross-origin writes denied'),403)
            if urlparse(self.path).path=='/api/import':
                return self.import_media()
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=2*1024*1024: raise ValueError('Invalid request size')
            if not self.headers.get('Content-Type','').startswith('application/json'): raise ValueError('JSON required')
            data=json.loads(self.rfile.read(size)); path=urlparse(self.path).path
            if path=='/api/import/start':
                name=str(data['name']).replace('\\','/').split('/')[-1]
                batch=str(data['batch']); size=int(data['size'])
                if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',batch): raise ValueError('Invalid import batch')
                if Path(name).suffix.lower() not in IMAGE_EXT|VIDEO_EXT: raise ValueError('Unsupported media type')
                if not 0<size<=20*1024**3: raise ValueError('File must be nonempty and at most 20 GB')
                folder=BASE/'tool_data/imports'/batch;folder.mkdir(parents=True,exist_ok=True)
                if shutil.disk_usage(folder).free<size+100*1024**2: raise ValueError('Insufficient local disk space')
                suffix=Path(name).suffix
                clean=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',Path(name).stem)[:120].rstrip(' .')
                token=uuid.uuid4().hex
                target=folder/(token[:8]+'_'+clean+suffix)
                partial=target.with_name(target.name+'.partial');partial.touch(exist_ok=False)
                with LOCK: IMPORTS[token]=dict(name=name,size=size,written=0,target=target,partial=partial)
                self.send_json(dict(token=token))
            elif path=='/api/import/chunk':
                with LOCK:
                    upload=IMPORTS[data['token']]
                    if int(data['offset'])!=upload['written']: raise ValueError('Incorrect chunk offset')
                    try: chunk=base64.b64decode(data['data'],validate=True)
                    except (ValueError,binascii.Error): raise ValueError('Invalid chunk data')
                    if not 0<len(chunk)<=512*1024 or upload['written']+len(chunk)>upload['size']: raise ValueError('Invalid chunk size')
                    with upload['partial'].open('ab') as f: f.write(chunk)
                    upload['written']+=len(chunk)
                    self.send_json(dict(written=upload['written']))
            elif path=='/api/import/finish':
                with LOCK:
                    upload=IMPORTS[data['token']]
                    if upload['written']!=upload['size']: raise ValueError('File transfer incomplete')
                    upload['partial'].replace(upload['target']);del IMPORTS[data['token']]
                p=upload['target']
                self.send_json(dict(path=p.relative_to(ROOT).as_posix(),folder=p.parent.relative_to(ROOT).as_posix(),
                                    name=upload['name'],kind='video' if p.suffix.lower() in VIDEO_EXT else 'image',reviewed=False,size=upload['size']))
            elif path=='/api/import/abort':
                with LOCK:
                    upload=IMPORTS.pop(data['token'],None)
                    if upload: upload['partial'].unlink(missing_ok=True)
                self.send_json(dict(aborted=True))
            elif path=='/api/predict':
                p=safe_path(data['path'],file=True)
                if p.suffix.lower() not in IMAGE_EXT: raise ValueError('Select an image')
                thresholds=confidence_settings(data.get('conf_by_class',data.get('conf',.25)))
                boxes,_=infer(str(p),thresholds,data.get('imgsz',640),data.get('position')); self.send_json(dict(boxes=boxes,reviewed=False,conf_by_class=thresholds))
            elif path=='/api/review':
                p=safe_path(data['path'],file=True); info=image_info(p)
                boxes=validate_boxes(data['boxes'],info['width'],info['height'])
                review=dict(path=p.relative_to(ROOT).as_posix(),width=info['width'],height=info['height'],boxes=boxes,reviewed=True,updated_at=time.time())
                with LOCK: atomic_json(BASE/'tool_data/reviews'/f'{file_id(p)}.json',review)
                self.send_json(dict(saved=True))
            elif path=='/api/jobs':
                paths=data['paths']
                if not isinstance(paths,list) or not 0<len(paths)<=20000: raise ValueError('Select media')
                for p in paths: safe_path(p,file=True)
                conf=confidence_settings(data.get('conf_by_class',data.get('conf',.25))); imgsz=int(data.get('imgsz',640))
                if imgsz not in (640,960,1280): raise ValueError('Invalid settings')
                position=position_settings(data.get('position'))
                job=dict(id=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6],state='queued',total=len(paths),done=0,errors=[],cancel_requested=False,conf_by_class=conf,imgsz=imgsz,position=position)
                with LOCK: JOBS[job['id']]=job
                job_update(job); POOL.submit(run_job,job,paths,conf,imgsz,position); self.send_json(job)
            elif path=='/api/cancel':
                with LOCK: job=JOBS[data['id']]; job_update(job,cancel_requested=True)
                self.send_json(dict(cancel_requested=True))
            elif path=='/api/export-reviewed':
                output=BASE/'tool_data/exports'/('reviewed_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
                (output/'images').mkdir(parents=True); (output/'labels').mkdir()
                records=[]
                with LOCK:
                    for p in (BASE/'tool_data/reviews').glob('*.json'):
                        r=json.loads(p.read_text(encoding='utf-8')); source=safe_path(r['path'],file=True)
                        boxes=validate_boxes(r['boxes'],r['width'],r['height']); stem=source.stem+'_'+file_id(source)
                        shutil.copy2(source,output/'images'/(stem+source.suffix))
                        (output/'labels'/f'{stem}.txt').write_text(yolo_text(boxes,r['width'],r['height']),encoding='utf-8')
                        records.append(r)
                atomic_json(output/'meta.json',dict(reviewed=True,images=len(records),classes=NAMES,records=records))
                self.send_json(dict(output=output.relative_to(ROOT).as_posix(),images=len(records)))
            else: self.send_json(dict(error='Not found'),404)
        except (ValueError,KeyError,FileNotFoundError) as exc: self.send_json(dict(error=str(exc)),400)
        except Exception as exc: self.send_json(dict(error=str(exc)),500)

    def import_media(self):
        """Stream explicitly chosen browser files to a local, independent copy."""
        query=parse_qs(urlparse(self.path).query)
        original_name=query.get('name',[''])[0].replace('\\','/').split('/')[-1]
        batch=query.get('batch',[''])[0]
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',batch): raise ValueError('Invalid import batch')
        if Path(original_name).suffix.lower() not in VIDEO_EXT|IMAGE_EXT: raise ValueError('Unsupported media type')
        size=int(self.headers.get('Content-Length','0'))
        if not 0<size<=20*1024**3: raise ValueError('File must be nonempty and at most 20 GB')
        if self.headers.get('Content-Type','').split(';')[0]!='application/octet-stream': raise ValueError('Binary media required')
        folder=BASE/'tool_data/imports'/batch
        folder.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(folder).free<size+100*1024**2: raise ValueError('Insufficient local disk space')
        # UUID prefix also avoids Windows device names and duplicate file collisions.
        filename=uuid.uuid4().hex[:8]+'_'+re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',original_name).rstrip(' .')
        filename=filename[:160-len(Path(filename).suffix)]+Path(filename).suffix if len(filename)>160 else filename
        target=folder/filename
        partial=folder/(filename+'.partial')
        remaining=size
        self.connection.settimeout(120)
        try:
            with partial.open('xb') as output:
                while remaining:
                    chunk=self.rfile.read(min(1024*1024,remaining))
                    if not chunk: raise ValueError('File transfer interrupted')
                    output.write(chunk);remaining-=len(chunk)
            partial.replace(target)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        self.send_json(dict(path=target.relative_to(ROOT).as_posix(),folder=folder.relative_to(ROOT).as_posix(),
                            name=original_name,kind='video' if target.suffix.lower() in VIDEO_EXT else 'image',reviewed=False,size=size))


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address=False
    def server_bind(self):
        if os.name=='nt': self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8765); args=parser.parse_args()
    for p in (BASE/'tool_data/jobs').glob('*.json'):
        j=json.loads(p.read_text(encoding='utf-8'))
        if j['state'] in ('queued','running'): j.update(state='interrupted',error='Server stopped before task completed')
        JOBS[j['id']]=j
    server=LocalServer(('127.0.0.1',args.port),Handler)
    print(f'Cone tool: http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    finally: server.server_close(); POOL.shutdown(wait=False,cancel_futures=True)


if __name__=='__main__': main()
