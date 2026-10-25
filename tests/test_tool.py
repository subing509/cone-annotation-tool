import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request

SOURCE=Path(__file__).resolve().parents[1]/'cone_project/tool_server.py'
spec=importlib.util.spec_from_file_location('tool_server',SOURCE)
app=importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class ToolTests(unittest.TestCase):
    def test_class_thresholds(self):
        self.assertEqual(app.confidence_settings(.3),dict(red=.3,blue=.3,yellow=.3))
        for bad in [float('nan'),0,1,{'red':.2}]:
            with self.assertRaises((ValueError,TypeError)):app.confidence_settings(bad)

    def test_positions(self):
        settings=app.position_settings()
        center=app.estimate_position({'xyxy':[450,200,550,300]},1000,600,settings)
        self.assertEqual(center['z_m'],2.25)
        self.assertEqual(center['x_m'],0)
        self.assertGreater(app.estimate_position({'xyxy':[600,200,700,300]},1000,600,settings)['x_m'],0)
        self.assertLess(app.estimate_position({'xyxy':[300,200,400,300]},1000,600,settings)['x_m'],0)
        self.assertFalse(app.estimate_position({'xyxy':[0,200,100,300]},1000,600,settings)['position_valid'])
        self.assertFalse(app.estimate_position({'xyxy':[450,200,550,300]},1000,600,app.position_settings({'enabled':False}))['position_valid'])

    def test_bounds_and_classes(self):
        for box in [{'class_id':3,'xyxy':[1,1,10,10]},{'class_id':0,'xyxy':[-1,1,10,10]},{'class_id':0,'xyxy':[1,1,float('nan'),10]}]:
            with self.assertRaises(ValueError):app.validate_boxes([box],100,100)
        with self.assertRaises(ValueError):app.safe_path('../../outside')

    def test_empty_install_http(self):
        original_base,original_root=app.BASE,app.ROOT
        with tempfile.TemporaryDirectory() as temp:
            app.ROOT=Path(temp);app.BASE=Path(temp)/'cone_project'
            (app.BASE/'web').mkdir(parents=True)
            (app.BASE/'web/index.html').write_text('<title>Cone tool</title>',encoding='utf-8')
            (app.ROOT/'JPEGImage').mkdir()
            server=app.LocalServer(('127.0.0.1',0),app.Handler)
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                base=f'http://127.0.0.1:{server.server_port}'
                with urllib.request.urlopen(base+'/api/model') as r:
                    self.assertFalse(json.load(r)['configured'])
                with urllib.request.urlopen(base+'/api/files?folder=JPEGImage') as r:
                    self.assertEqual(json.load(r)['total'],0)
                with urllib.request.urlopen(base+'/') as r:self.assertEqual(r.status,200)
            finally:
                server.shutdown();server.server_close()
                app.BASE,app.ROOT=original_base,original_root


if __name__=='__main__':unittest.main()
