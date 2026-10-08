import importlib.util
import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
import pandas as pd
root_dir=Path.cwd()
spec=importlib.util.spec_from_file_location('spie_title_smoke',root_dir/'Apps/SPIE/locust_lfp_gui.py')
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
import preparation_spike_analysis as analysis
folder=root_dir/'Analysis Outputs/current_title_validation'
source=folder/'sample_counts.csv'
pd.DataFrame([dict(epoch_label=e,channel='A-000',window_start_s=0,window_end_s=d,spike_count=r*d) for e,d,r in [('Baseline',1200,2),('Post 1',60,3)]]).to_csv(source,index=False)
root=tk.Tk()
root.withdraw()
app=m.LocustPipelineApp(root)
app.plot_input_var.set(str(source))
app.plot_out_dir_var.set(str(folder/'gui_exports'))
app.preparation_default_current.set('-100')
plot_spec=next(s for s in analysis.discover_plot_specs([source],[-100],'normalized') if s['kind']=='current')
app.preparation_plot_titles[plot_spec['plot_id']]='Custom GUI title: Post 1'
cmd,_=app._preparation_plot_command('normalized')
assert json.loads(cmd[cmd.index('--titles-json')+1])[plot_spec['plot_id']]=='Custom GUI title: Post 1'
root.destroy()
subprocess.run(cmd,check=True)
catalog=pd.read_csv(folder/'gui_exports/CSVs/plot_catalog.csv').set_index('plot_id')
assert catalog.loc[plot_spec['plot_id'],'title']=='Custom GUI title: Post 1'
print('GUI command to exported plot title: PASS')
print(folder/'gui_exports/Plots'/f"{plot_spec['plot_id']}.png")
