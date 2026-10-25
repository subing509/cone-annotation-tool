"""Windows-safe YOLO11 training, reporting and automatic held-out evaluation."""
from pathlib import Path
import argparse
import json
import os
import sys
import traceback
from datetime import datetime

BASE = Path(__file__).resolve().parent
for folder in ('.ultralytics/Ultralytics', '.matplotlib'):
    (BASE / folder).mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(BASE / '.ultralytics')
os.environ['MPLCONFIGDIR'] = str(BASE / '.matplotlib')
os.environ['YOLO_AUTOINSTALL'] = 'false'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO
    # Ultralytics AMP check resolves yolo11n.pt relative to the working directory.
    os.chdir(BASE / 'models')
    assert torch.cuda.is_available(), 'CUDA unavailable; refusing unintended CPU training.'
    name = 'smoke_yolo11n' if args.smoke else 'yolo11n_baseline'
    status_file = BASE / f'{name}_status.json'

    def status(state, **kw):
        payload = dict(state=state, time=datetime.now().astimezone().isoformat(), pid=os.getpid(), run=str(BASE / 'runs' / name), **kw)
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(status_file)

    config = dict(data=str(BASE / 'data' / 'data.yaml'), epochs=1 if args.smoke else 100,
                  imgsz=640, batch=8, device=0, workers=0 if args.smoke else 2,
                  project=str(BASE / 'runs'), name=name, seed=42, deterministic=True,
                  amp=True, cache=False, patience=25, save=True, save_period=10,
                  hsv_h=0.0, hsv_s=0.35, hsv_v=0.3, flipud=0.0, fliplr=0.5,
                  degrees=0.0, close_mosaic=10, plots=True, verbose=False,
                  fraction=0.025 if args.smoke else 1.0, exist_ok=args.smoke)
    try:
        status('initializing', gpu=torch.cuda.get_device_name(0), torch=torch.__version__, ultralytics=ultralytics.__version__, config=config)
        if args.resume:
            model = YOLO(str(BASE / 'runs' / name / 'weights' / 'last.pt'))
        else:
            model = YOLO(str(BASE / 'models' / 'yolo11n.pt'))
        def epoch_end(trainer):
            metrics = {str(k): float(v) for k,v in trainer.metrics.items()}
            status('training', epoch=trainer.epoch+1, total_epochs=trainer.epochs, metrics=metrics)
            print(f'\nEPOCH_COMPLETE {trainer.epoch+1}/{trainer.epochs} {metrics}', flush=True)
        model.add_callback('on_fit_epoch_end', epoch_end)
        model.train(resume=True) if args.resume else model.train(**config)
        if args.smoke:
            status('completed', stage='smoke')
            return
        best = BASE / 'runs' / name / 'weights' / 'best.pt'
        status('evaluating_test', weights=str(best))
        final = YOLO(str(best))
        metrics = final.val(data=config['data'], split='test', imgsz=640, batch=8, device=0, workers=2,
                            project=str(BASE / 'runs'), name='yolo11n_test', plots=True)
        report = dict(metrics=metrics.results_dict, classes=final.names,
                      per_class_map50=metrics.box.ap50.tolist(), per_class_map=metrics.box.ap.tolist(),
                      per_class_precision=metrics.box.p.tolist(), per_class_recall=metrics.box.r.tolist(),
                      class_indices=metrics.box.ap_class_index.tolist(), speed=metrics.speed,
                      weights=str(best), split='test', limitation='Unknown acquisition sources; ordered split with guards cannot guarantee independent scenes.')
        (BASE / 'test_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        status('completed', weights=str(best), test_metrics=metrics.results_dict)
    except BaseException as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
