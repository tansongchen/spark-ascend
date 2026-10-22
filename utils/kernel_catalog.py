"""Deliverable sources and their existing loader contracts; no source generation."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Kernel:
    source: str
    variant: str = 'stage_prefetch'
    description: str = ''
    manifest: dict = field(default_factory=dict)

    @property
    def library(self):
        return f'libdecode_layer_{self.variant}.so'


KERNELS = {
    'stage_prefetch': Kernel('decode_layer_stage_prefetch.asc', description='BF16 参考融合 decode'),
    'lossless_prefetch': Kernel('decode_layer_lossless_prefetch.asc', description='BF16 无损编码参考', manifest={
        'candidate': 'lossless_mlp_fast_kernels', 'weight_encoding': 'bf16_lossless12',
        'codec_base': 109, 'codec_repairs': 256, 'unchanged_abi': True,
        'unchanged_qkv_o_packing': True, 'cpu_codec': 'utils/bf16_codec.py',
    }),
}


def manifest_for(name):
    kernel = KERNELS[name]
    return {'candidate': name, 'workspace_bytes': 131072, 'target': 'dav-3510',
            'source': kernel.source, 'variant': kernel.variant, 'kernel': name,
            **kernel.manifest}
