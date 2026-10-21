"""Audit saved complete-request measurements without changing original evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def audit(path):
    result = json.loads(path.read_text())
    settings = result['settings']
    assert result['complete'] and not result['profiled'] and not result['validation_overhead']
    assert settings['npu'] != 0 and settings['warmup'] >= 1
    assert len(result['runs']) == 2 * settings['rounds']
    assert len(result['warmups']) == 2 * settings['warmup']
    for artifact in result['libraries'].values():
        assert hashlib.sha256(Path(artifact['library']).read_bytes()).hexdigest() == artifact['sha256']
    samples = []
    for line in path.with_suffix('.health.jsonl').read_text().splitlines():
        sample = json.loads(line)
        assert sample['returncode'] == 0
        sample['values'] = {k.strip(): float(v.strip()) for k, v in
                            (line.split(':', 1) for line in sample['common'].splitlines() if ':' in line)}
        samples.append(sample)
    rows = []
    previous_steps = 0
    for row in result['warmups'] + result['runs']:
        selection = row['worker_selection'][0]
        assert selection['decode_steps'] == previous_steps
        previous_steps += settings['output_tokens'] - 1
        assert selection['sha256'] == result['libraries'][row['mode']]['sha256']
        if 'worker_finished' in row:
            finished = row['worker_finished'][0]
            assert finished['decode_steps'] == previous_steps
            assert finished['library_sha256'] == selection['sha256']
            assert finished['ready'] and finished['layers'] == 36 and not finished['graph_enabled']
        if 'token_ids' not in row:
            continue
        assert len(row['token_ids']) == len(row['token_arrivals_s']) == settings['output_tokens']
        assert row['token_ids'] == result['runs'][0]['token_ids']
        assert hashlib.sha256(json.dumps(row['token_ids']).encode()).hexdigest() == row['token_sha256']
        arrivals = row['token_arrivals_s']
        assert all(a < b for a, b in zip(arrivals, arrivals[1:]))
        assert abs(row['tpot_ms'] - 1000 * (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1)) < 1e-9
        decode = [s['values'] for s in samples
                  if row['utc_start'] + row['ttft_ms']/1000 <= s['utc'] <= row['utc_end']]
        assert decode, 'No frequency samples in measured decode window'
        active = [s for s in decode if s['Aicore Usage Rate(%)'] > 0]
        clock_ok = bool(active) and all(s['Aicore curFreq(MHZ)'] == s['Aicore Freq(MHZ)'] for s in active)
        rows.append(dict(mode=row['mode'], request=row['request'], tpot_ms=row['tpot_ms'],
                         decode_samples=len(decode), frequency_MHz=dict(Counter(s['Aicore curFreq(MHZ)'] for s in decode)),
                         active_samples_at_rated_clock=clock_ok,
                         cooldown_seconds=row.get('cooldown_seconds', 0),
                         start_temperature_C=row.get('start_temperature_C'),
                         utilization_percent=[min(s['Aicore Usage Rate(%)'] for s in decode), max(s['Aicore Usage Rate(%)'] for s in decode)],
                         temperature_C=[min(s['Temperature(C)'] for s in decode), max(s['Temperature(C)'] for s in decode)]))
    assert result['worker_status'][0]['decode_steps'] == previous_steps
    assert result['worker_status'][0]['ready'] and result['worker_status'][0]['layers'] == 36
    clocks_ok = all(row['active_samples_at_rated_clock'] for row in rows)
    means = {mode: {field: statistics.mean(r[field] for r in result['runs'] if r['mode'] == mode)
                    for field in ('tpot_ms', 'ttft_ms', 'e2e_ms')} for mode in ('baseline', 'candidate')}
    for mode in means:
        for field, value in means[mode].items():
            assert abs(value - result['summary'][mode][field]) < 1e-9
    return dict(result=str(path), measurement_audit_passed=True, means=means,
                all_measured_active_samples_at_rated_clock=clocks_ok,
                tpot_reduction_percent=100*(1-means['candidate']['tpot_ms']/means['baseline']['tpot_ms']),
                timing_target_met=(settings['input_tokens'] == 32768 and settings['output_tokens'] == 4096
                                   and settings['rounds'] >= 3 and means['candidate']['tpot_ms'] <= 7.0 and clocks_ok),
                all_measured_output_tokens_equal=True, total_decode_steps=previous_steps, runs=rows,
                caveat='Frequency samples cover measured decode windows; sparse sampling is not proof of constant clock. Full numerical acceptance is recorded separately.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new audit output path')
    evidence = audit(args.result.resolve(strict=True))
    args.output.write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(evidence, indent=2))
