#!/usr/bin/env python3
"""驗證硬體加速（設定的「硬體加速」，worker 的 --accel）在這台電腦上能不能用、結果是否與 CPU 相同。

用安裝好的模型與同一個 Python（要有 numpy 與支援加速的 onnxruntime：onnxruntime-gpu、onnxruntime-migraphx、
onnxruntime-openvino 或 onnxruntime-ep-openvino），以與引擎相同的方式啟動 worker（-I -B -X utf8、stdio 協定），
CPU 與加速各跑一次同一批請求，比較：

- worker 就緒時回報的實際裝置與 provider（要求了加速卻是 cpu 就是不能用，原因在 stderr）
- 選字：首選與 CPU 相同的比例、分數最大差距；補標點：補出的標點與 CPU 相同的比例
- 往返延遲（平均、p50、p99、最大），以及超過引擎期限（選字 100 ms、補標點 150 ms）的次數

請求在本機依模型的字表產生（不含任何使用者文字，不連網）；也可以用 --requests 給錄下的選字請求（JSONL）、
--texts 給補標點文字（每行一段）。結果印在終端機；--json 另存一份給回報問題用（只有統計數字與環境，不含請求內容）。

  python3 accel_check.py --select <模型目錄>/select --punct <模型目錄>/punct --accel gpu
"""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
DEADLINE_MS = {'select': 100.0, 'punct': 150.0}
# 判定門檻：GPU 以 FP32 計算，應與 CPU 相同；NPU 以 fp16 計算，允許少數首選不同（量測約 99.8%）。
MIN_SAME = {'gpu': 0.995, 'npu': 0.99}
TONES = ['', 'ˊ', 'ˇ', 'ˋ']


def percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))]


def start(script, args, python):
    t = time.monotonic()
    p = subprocess.Popen([python, '-I', '-B', '-X', 'utf8', str(script)] + args, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')   # OpenVINO 在 Windows 以系統語系（cp950 等）印記錄
    line = p.stdout.readline()
    if not line:
        err = p.stderr.read()
        raise RuntimeError(f'worker 沒有就緒：{err.strip().splitlines()[-1] if err.strip() else "（沒有輸出）"}')
    ready = json.loads(line)
    return p, ready, time.monotonic() - t


def finish(p):
    p.stdin.close()
    p.wait(timeout=30)
    # stderr 只留 worker 自己印的說明（gpu unavailable、npu cannot compile…），不留 onnxruntime 的大量記錄。
    lines = [l for l in p.stderr.read().splitlines() if l and not l.startswith('[') and not l.startswith('\x1b')]
    return [l for l in lines if any(k in l for k in ('unavailable', 'disabled', 'cannot compile', 'using cpu'))][:5]


def exchange(p, request):
    t = time.perf_counter()
    p.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
    p.stdin.flush()
    reply = json.loads(p.stdout.readline())
    return reply, (time.perf_counter() - t) * 1000


def synthetic_select(vocab, manifest, count, seed):
    """依字表產生形狀像逐鍵輸入的請求：2–8 個音節、2–8 個候選（彼此只差一兩個字，共用前綴）、0–64 字前文。"""
    rnd = random.Random(seed)
    chars, bases = vocab['characters'][:4000], vocab['bases']
    seg = lambda t: {'text': t, 'log_score': -rnd.random() * 5, 'segments': [{'Start': 0, 'End': len(t), 'Word': t, 'Partial': False, 'Correction': False}]}
    out = []
    while len(out) < count:
        n, k = rnd.randint(2, 8), rnd.randint(2, 8)
        base = [rnd.choice(chars) for _ in range(n)]
        texts = {''.join(base)}
        while len(texts) < k:
            c = list(base)
            for _ in range(rnd.randint(1, 2)):
                c[rnd.randrange(n)] = rnd.choice(chars)
            texts.add(''.join(c))
        out.append({'context': ''.join(rnd.choice(chars) for _ in range(rnd.choice([0, 8, 32, 64]))),
                    'reading': [rnd.choice(bases) + rnd.choice(TONES) for _ in range(n)],
                    'candidates': [seg(t) for t in texts], 'allow_partial': bool(manifest.get('allow_partial', False))})
    return out


def synthetic_texts(vocab, count, seed):
    rnd = random.Random(seed)
    chars = vocab['characters'][:3000]
    return [''.join(rnd.choice(chars) for _ in range(rnd.choice([8, 16, 32, 64, 128, 200]))) for _ in range(count)]


def run_pair(kind, script, base_args, accel, requests, python, make_request, compare):
    """CPU 與加速各跑一次；回傳統計。"""
    result = {}
    for mode in ('cpu', accel):
        p, ready, startup = start(script, base_args + (['--accel', mode] if mode != 'cpu' else []), python)
        replies, lat = [], []
        for i, r in enumerate(requests):
            reply, ms = exchange(p, make_request(i, r))
            replies.append(reply)
            lat.append(ms)
        notes = finish(p)
        result[mode] = {'ready': ready, 'startup_s': round(startup, 2), 'replies': replies, 'latency': lat, 'notes': notes}
    same, total, worst = compare(result['cpu']['replies'], result[accel]['replies'])
    a = result[accel]
    stats = lambda lat: {'mean': round(sum(lat) / len(lat), 2), 'p50': round(percentile(lat, 50), 2),
                         'p99': round(percentile(lat, 99), 2), 'max': round(max(lat), 2),
                         'over_deadline': sum(x > DEADLINE_MS[kind] for x in lat)}
    return {'requested': accel, 'reported_accel': a['ready'].get('accel', 'cpu'), 'provider': a['ready'].get('accel_provider'),
            'startup_s': {'cpu': result['cpu']['startup_s'], accel: a['startup_s']},
            'same': same, 'compared': total, 'max_score_diff': worst,
            'latency_ms': {'cpu': stats(result['cpu']['latency']), accel: stats(a['latency'])},
            'worker_notes': a['notes']}


def compare_select(cpu, acc):
    same = total = 0
    worst = 0.0
    for a, b in zip(cpu, acc):
        if 'scores' not in a or 'scores' not in b:
            continue
        total += 1
        same += max(range(len(a['scores'])), key=a['scores'].__getitem__) == max(range(len(b['scores'])), key=b['scores'].__getitem__)
        worst = max(worst, max(abs(x - y) for x, y in zip(a['scores'], b['scores'])))
    return same, total, round(worst, 6)


def compare_punct(cpu, acc):
    pairs = [(a.get('edits'), b.get('edits')) for a, b in zip(cpu, acc) if 'edits' in a and 'edits' in b]
    return sum(a == b for a, b in pairs), len(pairs), None


def verdict(r, accel):
    """PASS：能用、結果與 CPU 相同、比較快；SLOWER：能用、正確但比較慢；DIFFERS：NPU 以產生的請求比較時差異超過門檻
    （見下）；FAIL：不能用、結果不同或超過引擎期限。"""
    if r['reported_accel'] != accel:
        return 'FAIL', f'要求 {accel}，worker 實際用 {r["reported_accel"]}（加速無法使用）'
    if r['compared'] == 0:
        return 'FAIL', '沒有可比較的結果'
    ratio = r['same'] / r['compared']
    over = r['latency_ms'][accel]['over_deadline']
    if over:
        return 'FAIL', f'{over} 次超過引擎期限'
    if ratio < MIN_SAME[accel]:
        if accel == 'npu' and r.get('synthetic'):
            # 產生的候選是隨機字，分數彼此很接近又很低，正是 fp16 最容易排錯的情況；逐鍵輸入的真實請求約 99.8% 相同。
            return 'DIFFERS', (f'與 CPU 相同 {ratio:.2%}（門檻 {MIN_SAME[accel]:.1%}）。NPU 以 fp16 計算，隨機產生的候選分數接近，'
                               '比真實輸入更容易排錯；要公平比較請用 --requests 給錄下的請求')
        return 'FAIL', f'與 CPU 相同只有 {ratio:.2%}（門檻 {MIN_SAME[accel]:.1%}）'
    speedup = r['latency_ms']['cpu']['mean'] / max(r['latency_ms'][accel]['mean'], 1e-9)
    if speedup < 1:
        return 'SLOWER', f'結果正確，但平均比 CPU 慢（{speedup:.2f} 倍）'
    return 'PASS', f'結果與 CPU 相同 {ratio:.2%}，平均快 {speedup:.2f} 倍'


def print_report(name, r, accel):
    v, why = verdict(r, accel)
    lc, la = r['latency_ms']['cpu'], r['latency_ms'][accel]
    print(f'\n== {name}：{v}（{why}）')
    print(f'   實際裝置 {r["reported_accel"]}，provider {r["provider"] or "—"}；就緒時間 CPU {r["startup_s"]["cpu"]} s、{accel} {r["startup_s"][accel]} s')
    print(f'   與 CPU 相同 {r["same"]}/{r["compared"]}' + (f'，分數最大差距 {r["max_score_diff"]}' if r['max_score_diff'] is not None else ''))
    print(f'   往返延遲 ms   CPU 平均 {lc["mean"]} p50 {lc["p50"]} p99 {lc["p99"]} 最大 {lc["max"]}')
    print(f'                {accel} 平均 {la["mean"]} p50 {la["p50"]} p99 {la["p99"]} 最大 {la["max"]}，超過期限 {la["over_deadline"]} 次')
    for note in r['worker_notes']:
        print(f'   worker：{note}')
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--select', type=Path, help='選字模型目錄（含 manifest.json、model.onnx、vocab.json）')
    p.add_argument('--punct', type=Path, help='補標點模型目錄')
    p.add_argument('--accel', choices=('gpu', 'npu'), default='gpu')
    p.add_argument('--count', type=int, default=2000, help='產生的選字請求數（補標點為其 1/5）')
    p.add_argument('--requests', type=Path, help='錄下的選字請求（JSONL，rerank.Request 格式）；取代產生的請求')
    p.add_argument('--texts', type=Path, help='補標點文字（每行一段，最多 256 字）；取代產生的文字')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--python', default=sys.executable, help='啟動 worker 的 Python（預設就是執行這支腳本的 Python）')
    p.add_argument('--json', type=Path, help='另存統計結果（不含請求內容）')
    a = p.parse_args()
    if not a.select and not a.punct:
        p.error('至少給 --select 或 --punct')
    if a.accel == 'npu' and not a.select:
        p.error('NPU 只用在選字，請給 --select')
    try:
        import onnxruntime
        ort_info = {'version': onnxruntime.__version__, 'providers': onnxruntime.get_available_providers()}
    except Exception as e:   # noqa: BLE001
        ort_info = {'error': str(e)}
    env = {'platform': platform.platform(), 'python': platform.python_version(), 'onnxruntime': ort_info}
    print(f'環境：{env["platform"]}，Python {env["python"]}，onnxruntime {ort_info.get("version", "?")}')
    print(f'可用 provider：{", ".join(ort_info.get("providers", [])) or ort_info.get("error")}')
    report, verdicts = {'environment': env, 'accel': a.accel}, []

    if a.select:
        m = json.loads((a.select / 'manifest.json').read_text(encoding='utf-8'))
        vocab = json.loads((a.select / 'vocab.json').read_text(encoding='utf-8'))
        if a.requests:
            reqs = [json.loads(l) for l in a.requests.read_text(encoding='utf-8').splitlines() if l.strip()][:a.count]
        else:
            reqs = synthetic_select(vocab, m, a.count, 1)
        sha, dic = m['files']['model.onnx'], m['dictionary_sha256']
        make = lambda i, r: dict(r, id=i, revision=i, model_sha256=sha, dictionary_sha256=dic)
        args = ['--package', str(a.select), '--dictionary', dic, '--threads', str(a.threads)]
        print(f'\n選字：{len(reqs)} 個請求，CPU 與 {a.accel} 各跑一次…', flush=True)
        r = run_pair('select', HERE / 'runtime' / 'sequence_runtime.py', args, a.accel, reqs, a.python, make, compare_select)
        r['synthetic'] = not a.requests
        verdicts.append(print_report('選字', r, a.accel))
        report['select'] = r

    if a.punct and a.accel == 'gpu':
        vocab = json.loads((a.punct / 'vocab.json').read_text(encoding='utf-8'))
        if a.texts:
            texts = [l.strip()[:256] for l in a.texts.read_text(encoding='utf-8').splitlines() if l.strip()]
        else:
            texts = synthetic_texts(vocab, max(50, a.count // 5), 2)
        args = ['--package', str(a.punct), '--threads', str(a.threads)]
        print(f'\n補標點：{len(texts)} 段文字，CPU 與 gpu 各跑一次…', flush=True)
        r = run_pair('punct', HERE / 'runtime' / 'punct_runtime.py', args, 'gpu', texts, a.python,
                     lambda i, t: {'id': i, 'text': t}, compare_punct)
        r['synthetic'] = not a.texts
        verdicts.append(print_report('補標點', r, 'gpu'))
        report['punct'] = r
    elif a.punct:
        print('\n補標點：不支援 NPU（照常用 CPU），略過。')

    if a.json:
        a.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n統計已存到 {a.json}')
    overall = next((v for v in ('FAIL', 'DIFFERS', 'SLOWER') if v in verdicts), 'PASS')
    print(f'\n總結：{overall}')
    return 0 if overall == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
