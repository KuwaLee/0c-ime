#!/usr/bin/env python3
"""序列模型（PF）的 ONNX 本機 stdio worker；協定同 encoder_runtime，另收外殼確認的前文。

以前綴樹一次評分全部候選（共用前綴與候選共同開頭），分數等同逐候選評分。

無 Torch、網路、學習或背景改字。前文只用最後 64 字；字表外的前文字元視為 UNK（同訓練），
候選含字表外字元或讀音不支援時整例回退。

mode="text"（單音節聲調容錯）：只用前文續寫評分 logP(候選｜前文)，不放讀音 token，
讀音只用來核對長度；必須有前文。規則同 SequenceModel 的 text task。
套件宣告 neutral_tolerant_last（一聲音節清單）時，讀音最後一個音節若在清單內，另以輕聲讀音評分，
逐候選取較大者（同一次 ONNX 呼叫的分支）。
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading

import numpy as np

# onnxruntime 在 import 時（建立全域 Env）就啟動遙測：Linux／macOS 連到外部伺服器並在個人目錄寫裝置識別碼，
# Windows 寫 ETW 給系統的診斷資料服務。這兩個變數只在那時讀取，必須在 import onnxruntime 之前設定；
# 宿主也會設（internal/neural 的 NoTelemetryEnv，說明在那裡）。
os.environ['ORT_DISABLE_TELEMETRY'] = '1'
os.environ['ORT_RUNNING_UNIT_TESTS'] = '1'

MAX_FRAME = 65536
SCHEMA = 'zuin-sequence-runtime-v1'
SCHEMA_TONE_MASK = 'zuin-sequence-runtime-v2'
CONTEXT_POLICY = 'confirmed_left_64'
TONES = {'ˊ': 2, 'ˇ': 3, 'ˋ': 4, '˙': 5}
TONE_POLICY_SCHEMA = 'zuin-tone-mask-v1'
MAX_CANDIDATES = 16   # 同 ime rerank.MaxCandidates：打出讀音 8 個＋輕聲擴充最多 8 個
ACCELS = ('cpu', 'npu', 'gpu')
# NPU（OpenVINO）每遇到新的輸入形狀就要編譯一次（冷 ~1 s，驅動磁碟快取 ~0.2 s），遠超過宿主的時限，
# 所以 NPU 只收補齊後的固定形狀：L 補到 NPU_LENGTHS、proj_pos 補到 NPU_PROJ、target 攤平成單欄後補到 NPU_ROWS 列，
# 共 6 個形狀，啟動後在背景全部編譯。超出的請求走 CPU。
# 補齊不改變分數：補的 token 只注意自己、真實 token 看不到它們；proj_pos 與 target 多的列重複第 0 項，輸出再截掉；
# 候選分數是各字 log 機率的總和，攤平成單欄後在這裡加總，等於原本的 [k, n] 輸出。
# proj_pos 上限 24：每個位置要對整個字表（約 2.8 萬字）做 log_softmax，Core Ultra 200V 的 NPU 片上記憶體（CMX）
# 放不下 32 個位置，編譯失敗。逐鍵輸入的請求（前文 64 字）約 78% 在 24 以內，L 全部在 256 以內、列數 99.9% 在 256 以內。
NPU_LENGTHS = (128, 256)
NPU_PROJ = (8, 16, 24)
NPU_ROWS = 256
NPU_SHAPES = tuple((L, u) for L in NPU_LENGTHS for u in NPU_PROJ)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def syllable(y, bases):
    if not isinstance(y, str) or not y:
        raise ValueError('invalid_reading')
    tone = TONES.get(y[-1], 1)
    base = y[:-1] if y[-1] in TONES else y
    if base not in bases:
        raise ValueError('unknown_base')
    return bases[base], tone


def validate_tone_known(tone_known, count, allow_unknown_tones):
    """省略遮罩沿用舊協定：沒有聲調符號的讀音是一聲，不是未知。"""
    if tone_known is None:
        return [True] * count
    if not isinstance(tone_known, list) or len(tone_known) != count or any(type(x) is not bool for x in tone_known):
        raise ValueError('invalid_tone_known')
    if not allow_unknown_tones and not all(tone_known):
        raise ValueError('unsupported_unknown_tone')
    return tone_known


def prepare(context, reading, candidates, maps, allow_partial=False, text_only=False, neutral_last=False,
            tone_known=None, allow_unknown_tones=False):
    """前綴樹輸入：[BOS] 前文 [SEP] 讀音 [OUT] 只出現一次，候選共同開頭共用節點。

    每個節點只注意前綴與自己的祖先，位置編號同單獨評分，因此分數等同
    SequenceModel.task_logits（phonetic）逐候選評分；proj_pos 是需要輸出分布的唯一位置。
    text_only=True 時省略讀音 token（text task），只支援單音節且需有前文。
    neutral_last=True 且最後一個音節不是輕聲時，另加一支「最後音節改輕聲」的分支（共用前文與前面讀音）；
    target 列依分支排列（每支 len(candidates) 列），由呼叫端逐候選取最大值。
    tone_known=False 的音節只在新套件明確支援時以 tone id 0 編碼，保留聲韻母；
    全批候選共用同一遮罩。未知末音節不另建輕聲分支。
    """
    bases, chars = maps
    n = len(reading)
    known = validate_tone_known(tone_known, n, allow_unknown_tones)
    if not (n == 1 if text_only else 2 <= n <= 32) or not 2 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError('unsupported_shape')
    if len({c['text'] for c in candidates}) != len(candidates):
        raise ValueError('duplicate_candidates')
    if not isinstance(context, str) or (text_only and not context):
        raise ValueError('invalid_context')
    encoded = []
    for candidate in candidates:
        text = candidate['text']
        if len(text) != n or not math.isfinite(candidate['log_score']):
            raise ValueError('candidate_alignment_or_score')
        end = 0
        for seg in candidate['segments']:
            if seg['Start'] != end or not end < seg['End'] <= n or (seg['Partial'] and not allow_partial) \
                    or seg['Correction'] or seg['Word'] != text[end:seg['End']]:
                raise ValueError('unsupported_segment')
            end = seg['End']
        if end != n:
            raise ValueError('incomplete_segments')
        try:
            encoded.append([chars[c] for c in text])
        except KeyError as e:
            raise ValueError('unknown_character') from e
    sounds = [] if text_only else [syllable(y, bases) for y in reading]
    if not text_only:
        sounds = [(base, tone if keep else 0) for (base, tone), keep in zip(sounds, known)]
    # 讀音最後一個音節之前共用；最後一個音節（與 OUT）依變體分支，各自接一棵候選樹。
    branches = [[(0, 0)]] if text_only else [[sounds[-1], (0, 0)]]
    if neutral_last and not text_only and known[-1] and sounds[-1][1] != 5:
        branches.append([(sounds[-1][0], 5), (0, 0)])
    head = [1] + [chars.get(c, 4) for c in context[-64:]] + [2]
    shared = sounds[:-1]
    ids = head + [0] * len(shared)
    base_ids = [0] * len(head) + [b for b, _ in shared]
    tones = [0] * len(head) + [x for _, x in shared]
    types = [0] * len(head) + [1] * len(shared)
    S = len(ids); positions = list(range(S)); parent = {}; nodes = []
    for branch in branches:
        prev = None
        for k, (b, t) in enumerate(branch):
            if prev is not None:
                parent[len(ids)] = prev
            prev = len(ids)
            last = k == len(branch) - 1
            ids.append(3 if last else 0); base_ids.append(b); tones.append(t); types.append(0 if last else 1); positions.append(S + k)
        P = S + len(branch)   # OUT 之後第一個候選字的位置
        node = {(): prev}
        for cids in encoded:
            for i in range(1, n):
                key = tuple(cids[:i])
                if key not in node:
                    node[key] = len(ids); parent[len(ids)] = node[key[:-1]]
                    ids.append(cids[i - 1]); base_ids.append(0); tones.append(0); types.append(2); positions.append(P + i - 1)
        nodes.append(node)
    L = len(ids)
    mask = np.zeros((L, L), dtype=bool); mask[:S, :S] = np.tril(np.ones((S, S), dtype=bool))
    for q in range(S, L):
        mask[q, :S] = True; a = q
        while a is not None:
            mask[q, a] = True; a = parent.get(a)
    proj = sorted({v for node in nodes for v in node.values()}); index = {p: i for i, p in enumerate(proj)}
    target_node = np.array([[index[node[tuple(c[:i])]] for i in range(n)] for node in nodes for c in encoded], dtype=np.int64)
    target_ids = np.array([[x - 6 for x in c] for _ in nodes for c in encoded], dtype=np.int64)
    row = lambda x: np.array([x], dtype=np.int64)
    return {'ids': row(ids), 'bases': row(base_ids), 'tones': row(tones), 'types': row(types), 'positions': row(positions),
            'mask': mask[None, None], 'proj_pos': np.array(proj, dtype=np.int64), 'target_node': target_node, 'target_ids': target_ids}


def npu_shape(inputs):
    """回傳這次輸入補齊後的 (L, proj_pos 長度)；超出 NPU 形狀範圍時回 None（改用 CPU）。"""
    L, u = inputs['ids'].shape[1], inputs['proj_pos'].shape[0]
    if inputs['target_node'].size > NPU_ROWS:
        return None
    return next(((b, p) for b, p in NPU_SHAPES if b >= L and p >= u), None)


def pad_for_npu(inputs, shape):
    """把 prepare() 的輸入補到固定形狀 shape=(L, proj_pos 長度)、NPU_ROWS 列；分數由 unpad_scores 還原。"""
    length, proj_len = shape
    L = inputs['ids'].shape[1]
    out = {}
    for key in ('ids', 'bases', 'tones', 'types', 'positions'):
        out[key] = np.zeros((1, length), dtype=np.int64); out[key][0, :L] = inputs[key][0]
    mask = np.zeros((length, length), dtype=bool); mask[:L, :L] = inputs['mask'][0, 0]
    pad = np.arange(L, length); mask[pad, pad] = True   # 補的 token 只注意自己，避免整列遮掉
    out['mask'] = mask[None, None]
    proj = inputs['proj_pos']; out['proj_pos'] = np.concatenate([proj, np.full(proj_len - proj.size, proj[0], dtype=np.int64)])
    node, ids = inputs['target_node'].reshape(-1, 1), inputs['target_ids'].reshape(-1, 1)
    extra = NPU_ROWS - node.shape[0]
    out['target_node'] = np.concatenate([node, np.repeat(node[:1], extra, 0)])
    out['target_ids'] = np.concatenate([ids, np.repeat(ids[:1], extra, 0)])
    return out


def unpad_scores(raw, inputs):
    k, n = inputs['target_node'].shape
    return raw[:k * n].reshape(k, n).sum(1)


class NPU:
    """OpenVINO NPU session 與背景編譯。

    只有編譯過的形狀才走 NPU；沒編譯過的這一次照舊用 CPU，請求永遠不等編譯。形狀由 warm() 排進背景執行緒編譯。
    session 一次只給一個執行緒用（lock）；背景編譯占著時前景改用 CPU。
    某個形狀編譯失敗只停用那個形狀；已編譯的形狀執行出錯則停用 NPU，之後全部走 CPU。
    """

    def __init__(self, ort, path):
        options = ort.SessionOptions()
        # OpenVINO EP 在 NPU 編譯失敗時預設靜默改用 OpenVINO 的 CPU（比 onnxruntime 的 CPU 慢），get_providers() 也看不出來；
        # 關掉這個退路，失敗才會丟例外，由這裡改用 onnxruntime 的 CPU session。
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        session = ort.InferenceSession(path, sess_options=options, providers=[('OpenVINOExecutionProvider', {'device_type': 'NPU'})])
        # onnxruntime 建不起 OpenVINO EP（缺套件、沒有 NPU、沒有裝置權限）時只印警告並退回 CPU。
        if session.get_providers()[0] != 'OpenVINOExecutionProvider':
            raise RuntimeError('openvino_npu_unavailable')
        self.session, self.lock = session, threading.Lock()
        self.ready, self.pending, self.unsupported, self.failed = set(), set(), set(), False
        self.jobs = queue.Queue()
        threading.Thread(target=self._compile_loop, daemon=True).start()

    def warm(self, inputs, shape):
        """背景編譯這個形狀（把呼叫端給的樣本補齊；樣本只用字表，不含使用者文字）。"""
        if shape in self.ready or shape in self.pending or shape in self.unsupported or self.failed:
            return
        self.pending.add(shape)
        self.jobs.put((shape, pad_for_npu(inputs, shape)))

    def _compile_loop(self):
        while True:
            shape, padded = self.jobs.get()
            try:
                with self.lock:
                    raw = self.session.run(['scores'], padded)[0]
                if not np.isfinite(raw).all():
                    raise ValueError('nonfinite_neural_score')
                self.ready.add(shape)
            except Exception as e:   # noqa: BLE001 — 只停用這個形狀，CPU 照常
                self.unsupported.add(shape)
                print(f'npu cannot compile shape {shape}, using cpu: {str(e).splitlines()[0][:200]}', file=sys.stderr, flush=True)
            finally:
                self.pending.discard(shape)

    def disable(self, error):
        if not self.failed:
            self.failed = True
            print(f'npu disabled, using cpu: {error}', file=sys.stderr, flush=True)

    def run(self, inputs):
        """編譯過且 session 空著時回傳分數，否則回 None 讓呼叫端用 CPU。"""
        if self.failed:
            return None
        shape = npu_shape(inputs)
        if shape not in self.ready or not self.lock.acquire(blocking=False):
            return None
        try:
            raw = self.session.run(['scores'], pad_for_npu(inputs, shape))[0]
        except Exception as e:   # noqa: BLE001
            self.disable(e)
            return None
        finally:
            self.lock.release()
        return unpad_scores(raw, inputs)


def add_openvino_dll_dir():
    """Windows：onnxruntime-openvino 的 provider DLL 依賴 openvino 套件的 DLL（openvino\\libs），不在預設搜尋路徑。

    onnxruntime 附的 tools.add_openvino_win_libs 找不到時會 sys.exit，這裡改成找不到就不做事（建立 session 失敗後照舊用 CPU）。
    只找套件目錄、不 import openvino。其他平台的 wheel 自己帶 rpath，不必處理。
    """
    if sys.platform != 'win32':
        return
    import importlib.util
    spec = importlib.util.find_spec('openvino')
    for root in (spec.submodule_search_locations or []) if spec else []:
        libs = os.path.join(root, 'libs')
        if os.path.isdir(libs):
            os.add_dll_directory(libs)
            os.environ['PATH'] = libs + os.pathsep + os.environ.get('PATH', '')
            return


class GPU:
    """OpenVINO Intel GPU session（FP32）。

    GPU 收動態形狀，不必補齊：新形狀第一次約數十 ms（量測最多 46 ms），之後約 7 ms。FP16 不能用：
    遮罩以 float32 最小值相加，fp16 溢位成 inf／NaN。執行出錯或分數非有限值時這一次改用 CPU，
    連續出錯 GPU_MAX_ERRORS 次即停用。
    """
    GPU_MAX_ERRORS = 3

    def __init__(self, ort, path):
        options = ort.SessionOptions()
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        session = ort.InferenceSession(path, sess_options=options,
                                       providers=[('OpenVINOExecutionProvider', {'device_type': 'GPU', 'precision': 'FP32'})])
        if session.get_providers()[0] != 'OpenVINOExecutionProvider':
            raise RuntimeError('openvino_gpu_unavailable')
        self.session, self.errors = session, 0

    def run(self, inputs):
        if self.errors >= self.GPU_MAX_ERRORS:
            return None
        try:
            raw = self.session.run(['scores'], inputs)[0]
            if not np.isfinite(raw).all():
                raise ValueError('nonfinite_neural_score')
        except Exception as e:   # noqa: BLE001 — 這一次改用 CPU
            self.errors += 1
            if self.errors == self.GPU_MAX_ERRORS:
                print(f'gpu disabled, using cpu: {str(e).splitlines()[0][:200]}', file=sys.stderr, flush=True)
            return None
        self.errors = 0
        return raw


class Runtime:
    def __init__(self, package, dictionary, threads=2, accel='cpu'):
        import onnxruntime as ort
        if hasattr(ort, 'disable_telemetry_events'):   # 再保險一次：之後的 session 與推論事件也不記
            ort.disable_telemetry_events()
        package = Path(package)
        self.manifest = m = json.loads((package / 'manifest.json').read_text())
        if m['schema'] not in (SCHEMA, SCHEMA_TONE_MASK) or m['context_policy'] != CONTEXT_POLICY \
                or m['dictionary_sha256'] != dictionary:
            raise ValueError('package_contract_mismatch')
        self.allow_unknown_tones = m['schema'] == SCHEMA_TONE_MASK
        if not self.allow_unknown_tones and 'tone_policy' in m:
            raise ValueError('unsupported_tone_policy')
        if self.allow_unknown_tones:
            policy = m.get('tone_policy')
            if not isinstance(policy, dict) or policy.get('schema') != TONE_POLICY_SCHEMA \
                    or type(policy.get('unknown_tone_id')) is not int or policy['unknown_tone_id'] != 0:
                raise ValueError('unsupported_tone_policy')
        for name in ('model.onnx', 'vocab.json'):
            if sha(package / name) != m['files'][name]:
                raise ValueError('package_hash_mismatch')
        vocab = json.loads((package / 'vocab.json').read_text())
        if vocab['dictionary_sha256'] != dictionary:
            raise ValueError('vocabulary_dictionary_mismatch')
        self.maps = ({x: i + 2 for i, x in enumerate(vocab['bases'])}, {x: i + 8 for i, x in enumerate(vocab['characters'])})
        self.vocab = vocab
        options = ort.SessionOptions(); options.intra_op_num_threads = threads; options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(package / 'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        self.accel = None   # 加速裝置（NPU 或 GPU）；None 時只用 CPU
        if accel != 'cpu':
            try:
                add_openvino_dll_dir()
                self.accel =(NPU if accel == 'npu' else GPU)(ort, str(package / 'model.onnx'))
            except Exception as e:   # noqa: BLE001 — 裝置無法使用時照舊只用 CPU
                print(f'{accel} unavailable, using cpu: {e}', file=sys.stderr, flush=True)
        self.model_sha = m['files']['model.onnx']; self.dictionary = dictionary
        self.allow_partial = bool(m.get('allow_partial', False))
        self.text_mode = float(m.get('text_weight', 0)) > 0
        # 句尾助詞常被打成一聲：最後一個音節是清單中的一聲音節時，同時以輕聲評分取較大者。
        # 套件宣告 neutral_expand_penalty 時，宿主會併入輕聲讀音的候選，清單基底的任何非輕聲聲調都另以輕聲評分。
        self.neutral_tolerant = frozenset(m.get('neutral_tolerant_last', []))
        self.neutral_expand = float(m.get('neutral_expand_penalty', 0)) > 0
        self.neutral_bases = frozenset(y[:-1] if y and y[-1] in TONES else y for y in self.neutral_tolerant)

    def score(self, request):
        if request['dictionary_sha256'] != self.dictionary or request['model_sha256'] != self.model_sha:
            raise ValueError('request_version_mismatch')
        if bool(request.get('allow_partial', False)) != self.allow_partial:
            raise ValueError('candidate_policy_mismatch')
        mode = request.get('mode', '')
        if mode not in ('', 'text') or (mode == 'text' and not self.text_mode):
            raise ValueError('unsupported_mode')
        reading, count = request['reading'], len(request['candidates'])
        # JSON null 不是省略欄位；明確提供時只接受與音節逐項對齊的布林陣列。
        if 'tone_known' in request and request['tone_known'] is None:
            raise ValueError('invalid_tone_known')
        last = reading[-1] if reading and isinstance(reading[-1], str) else ''
        if self.neutral_expand:
            neutral = mode == '' and bool(last) and last[-1] != '˙' and (last[:-1] if last[-1] in TONES else last) in self.neutral_bases
        else:
            neutral = mode == '' and last in self.neutral_tolerant
        inputs = prepare(request.get('context', ''), reading, request['candidates'], self.maps, self.allow_partial,
                         text_only=mode == 'text', neutral_last=neutral, tone_known=request.get('tone_known'),
                         allow_unknown_tones=self.allow_unknown_tones)
        raw = self.accel.run(inputs) if self.accel else None
        if raw is None:
            raw = self.session.run(['scores'], inputs)[0]
        raw = raw.astype(float)
        if raw.shape[0] % count or not np.isfinite(raw).all():
            raise ValueError('nonfinite_neural_score')
        return raw.reshape(-1, count).max(0).tolist()   # 各讀音分支逐候選取最大


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True); p.add_argument('--dictionary', required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--accel', choices=ACCELS, default='cpu',
                   help='npu：已編譯形狀走 OpenVINO NPU；gpu：OpenVINO Intel GPU（FP32）；裝置無法使用時走 CPU')
    a = p.parse_args()
    if not 1 <= a.threads <= 8:
        p.error('threads must be between 1 and 8')
    runtime = Runtime(a.package, a.dictionary, a.threads, a.accel)
    # 暖機只用字表，不含新聞、私人輸入或評測資料。
    chars = runtime.vocab['characters'][8:12]; base = runtime.vocab['bases'][0]
    seg = lambda t: {'text': t, 'log_score': 0., 'segments': [{'Start': 0, 'End': 2, 'Word': t, 'Partial': False, 'Correction': False}]}
    warm = {'dictionary_sha256': a.dictionary, 'model_sha256': runtime.model_sha, 'context': ''.join(chars),
            'reading': [base] * 2, 'candidates': [seg(chars[0] + chars[1]), seg(chars[2] + chars[3])],
            'allow_partial': runtime.allow_partial}
    runtime.score(warm)
    if isinstance(runtime.accel, NPU):
        # 全部 6 個形狀在背景編譯（同一份字表樣本可補到任何形狀）：冷編譯每個約 1 s，之後有驅動的磁碟快取，約 0.2 s。
        sample = prepare(warm['context'], warm['reading'], warm['candidates'], runtime.maps, runtime.allow_partial)
        for shape in NPU_SHAPES:
            runtime.accel.warm(sample, shape)
    ready = {'ready': True, 'schema': runtime.manifest['schema'], 'model_sha256': runtime.model_sha,
             'dictionary_sha256': a.dictionary, 'allow_partial': runtime.allow_partial}
    if runtime.allow_unknown_tones:
        ready['tone_policy'] = runtime.manifest['tone_policy']
    # 實際使用的裝置：要求 gpu／npu 但裝置無法使用時是 cpu（宿主的 health 回報 selection_accel）。
    ready['accel'] = 'npu' if isinstance(runtime.accel, NPU) else 'gpu' if isinstance(runtime.accel, GPU) else 'cpu'
    print(json.dumps(ready), flush=True)
    while True:
        line = sys.stdin.buffer.readline(MAX_FRAME + 1)
        if not line:
            return
        if len(line) > MAX_FRAME or not line.endswith(b'\n'):
            raise ValueError('request_frame_limit')
        r = json.loads(line)
        reply = {'id': r['id'], 'revision': r['revision'], 'model_sha256': runtime.model_sha, 'dictionary_sha256': a.dictionary}
        try:
            reply['scores'] = runtime.score(r)
        except (ValueError, KeyError, TypeError):
            reply['fallback'] = 'unsupported_request'
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
