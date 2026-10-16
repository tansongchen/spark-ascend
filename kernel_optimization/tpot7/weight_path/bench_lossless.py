"""Lossless MLP packing plus unchanged single-layer CPU reference and AB/BA."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
s=(ROOT/'kernel_optimization/tpot7/weight_path/bench.py').read_text()
s=s.replace('ROOT = Path(__file__).resolve().parents[3]',f'ROOT=Path({str(ROOT)!r})')
needle="samples=[]"
assert s.count(needle)==1
s=s.replace(needle,"""
s=s.replace("s=s.replace('inputs[4]=tile_weights(inputs[4],8)','inputs[4]=tile_weights(inputs[4],args.up_tile)')", "s=s.replace('inputs[4]=tile_weights(inputs[4],8)', 'from kernel_optimization.tpot7.lossless.codec import pack_weight; inputs[4]=pack_weight(tile_weights(inputs[4],8),2560,8)')")
s=s.replace("s=s.replace('inputs[5]=tile_weights(inputs[5],4)','inputs[5]=tile_weights(inputs[5],args.down_tile)')", "s=s.replace('inputs[5]=tile_weights(inputs[5],4)', 'inputs[5]=pack_weight(tile_weights(inputs[5],4),10240,4)')")
if os.environ.get('BASELINE_LOSSLESS')=='1':
    s=s.replace('b1=tile_weights(w1,8).npu()', 'b1=dev[4]')
    s=s.replace("b2=tile_weights(w['mlp.down_proj.weight'],4).npu()", 'b2=dev[5]')
samples=[]
""")
assert int(os.environ.get('WEIGHT_ROTATIONS','1'))==1,'This wrapper retains separate baseline/candidate layouts'
exec(compile(s,str(ROOT/'kernel_optimization/tpot7/weight_path/bench.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'kernel_optimization/tpot7/weight_path/bench.py')})
