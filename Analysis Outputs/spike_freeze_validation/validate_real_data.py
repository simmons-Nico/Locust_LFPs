import importlib.util
import sys
import time
from pathlib import Path
root=Path.cwd()
spec=importlib.util.spec_from_file_location('spie_validation', root/'Apps/SPIE/locust_lfp_gui.py')
gui=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=gui
spec.loader.exec_module(gui)
paths=sorted(Path('C:/Users/simmons/Desktop/Exploring PSDs').glob('*.csv'))
labels={str(p): ('Baseline' if 'Baseline' in p.name else 'Post') for p in paths}
out=root/'Analysis Outputs/spike_freeze_validation/processed'
request=gui.SpikePipelineRequest([str(p) for p in paths],labels,str(out),60.0,'both',None,[])
start=time.perf_counter()
result=gui.run_shared_spike_pipeline(request)
print('ELAPSED_SECONDS',time.perf_counter()-start,flush=True)
print(result['spike_count_csv'],flush=True)
