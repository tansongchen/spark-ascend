"""Existing CPU reference plus the candidate's seven-way 384-aligned partition."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
s=(ROOT/'kernel_validation/validate.py').read_text().replace('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(ROOT)!r})')
s=s.replace('    width=5120 if rd==64 else 128','    width=((args.length+2687)//2688)*384 if rd==64 else 128')
exec(compile(s,str(ROOT/'kernel_validation/validate.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'kernel_validation/validate.py')})
