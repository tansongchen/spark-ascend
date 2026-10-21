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
    'stage_prefetch': Kernel('decode_layer_stage_prefetch.asc', description='默认 BF16 融合 decode'),
    'lossless_prefetch': Kernel('decode_layer_lossless_prefetch.asc', description='已验收 BF16 无损编码', manifest={
        'candidate': 'lossless_mlp_fast_kernels', 'weight_encoding': 'bf16_lossless12',
        'codec_base': 109, 'codec_repairs': 256, 'unchanged_abi': True,
        'unchanged_qkv_o_packing': True, 'cpu_codec': 'utils/bf16_codec.py',
    }),
    'w4a16_v10b': Kernel('decode_layer_w4a16_v10b.asc', 'w4a16', '当前最快 W4A16 v10b；整模型 logits 未通过既定门限', {
        'candidate': 'w4a16_direct_gemv', 'weight_encoding': 'spark_w4a16_tile_g128_v1',
        'attention_partials': 'bfloat16', 'swa_part_width': 128, 'full_part_width': 5120,
        'direct_gemv': True, 'hardware_convert': True, 'compact_convert': True,
        'persistent_full_bf16_weights': False, 'diagnostic_stamps': False,
    }),
}


def manifest_for(name):
    kernel = KERNELS[name]
    return {'candidate': name, 'workspace_bytes': 131072, 'target': 'dav-3510',
            'source': kernel.source, 'variant': kernel.variant, 'kernel': name,
            **kernel.manifest}
