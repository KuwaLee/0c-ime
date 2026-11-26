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
import time

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
# GPU 每次最多等這麼久（秒），沒等到就這一次用 CPU：宿主期限 100 ms，CPU 評分在慢的機器上也要留時間。
GPU_BUDGET = 0.03


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


# NPU 的 provider，依序試（Runtime 只在 onnxruntime 有這個 provider 時才試）：
# - openvino（Intel NPU，只有舊的 onnxruntime-openvino，不用 plugin，原因見 NPU.__init__）：一個 session 收動態形狀，
#   每個新形狀在第一次執行時編譯；關掉 CPU 退路（編譯失敗時 OpenVINO 預設靜默改用自己的 CPU，比 onnxruntime 的 CPU
#   慢），失敗才丟例外。
# - vitisai（AMD Ryzen AI，實驗性，未在 AMD 硬體上驗證）：NPU 編譯器要固定形狀，所以每個形狀各開一個 session，
#   以 free dimension override 把 L、u、k、n 固定；它本來就把 NPU 不支援的運算交給 CPU，不關 CPU 退路。
#   編譯結果放在 cache_dir（依模型雜湊分開）；部署版的 VitisAI EP 不能現場編譯 BF16，要裝完整的 Ryzen AI Software，
#   設定檔由 ZC_VITISAI_CONFIG 或 RYZEN_AI_INSTALLATION_PATH 底下的 vaip_config.json 提供。
NPU_PROVIDERS = (
    ('openvino', 'OpenVINOExecutionProvider'),
    ('vitisai', 'VitisAIExecutionProvider'),
)


def accel_cache_dir(name):
    """加速裝置的編譯快取：Windows 在 %LOCALAPPDATA%\\Zuin\\cache，其他在 $XDG_CACHE_HOME/0c（或 ~/.cache/0c）。"""
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'Zuin' / 'cache'
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / '0c'
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def vitisai_config():
    """VitisAI 的編譯設定檔：ZC_VITISAI_CONFIG，或 Ryzen AI Software 安裝目錄裡的 vaip_config.json；沒有時為 None。"""
    explicit = os.environ.get('ZC_VITISAI_CONFIG')
    if explicit and os.path.isfile(explicit):
        return explicit
    root = os.environ.get('RYZEN_AI_INSTALLATION_PATH')
    if root and os.path.isdir(root):
        found = sorted(Path(root).rglob('vaip_config.json'))
        if found:
            return str(found[0])
    return None


class NPU:
    """NPU session 與背景編譯（Intel 走 OpenVINO、AMD 走 VitisAI，見 NPU_PROVIDERS）。

    只有編譯過的形狀才走 NPU；沒編譯過的這一次照舊用 CPU，請求永遠不等編譯。形狀由 warm() 排進背景執行緒編譯。
    session 一次只給一個執行緒用（lock）；背景編譯占著時前景改用 CPU。
    某個形狀編譯失敗只停用那個形狀；已編譯的形狀執行出錯則停用 NPU，之後全部走 CPU。
    """

    def __init__(self, ort, path, model_sha):
        self.ort, self.path, self.model_sha = ort, path, model_sha
        available = set(ort.get_available_providers())
        errors = []
        for name, provider in NPU_PROVIDERS:
            # 不用 OpenVINO 的 plugin（onnxruntime-ep-openvino 1.8.1 實測）：建 session 時就編譯、NPU 不收動態形狀，只能
            # 每個形狀各開一個，而建 session 時不放開 GIL，每個形狀約 6 s 前景完全停住、超過期限；編譯快取讀回時又因 bool 的
            # mask 報「Type mismatch」而失敗。所以 Intel NPU 只支援舊的 onnxruntime-openvino（GPU 兩種都可以）。
            if provider not in available:
                continue
            try:
                self.name, self.provider = name, provider
                # openvino：一個動態形狀的 session 先建好；vitisai：每個形狀在背景編譯時各建一個。
                self.shared = self._open_retry(None) if name == 'openvino' else None
                break
            except Exception as e:   # noqa: BLE001 — 試下一個
                errors.append(f'{name}: {str(e).splitlines()[0][:120]}')
        else:
            raise RuntimeError('no_npu_provider' + (' (' + '; '.join(errors) + ')' if errors else ''))
        self.sessions, self.lock = {}, threading.Lock()
        self.ready, self.pending, self.unsupported, self.failed = set(), set(), set(), False
        self.jobs = queue.Queue()
        threading.Thread(target=self._compile_loop, daemon=True).start()

    def _open_retry(self, shape, attempts=8, wait=0.5):
        """Intel NPU 在上一個用 OpenVINO 的程序剛結束時有 0.5–1.5 s 開不起來（Windows 實測；引擎交接重啟正好碰上），
        稍等重試，最多約 3.5 s（還沒回報就緒，請求照常走統計引擎）。"""
        for i in range(attempts):
            try:
                return self._session(shape)
            except Exception:   # noqa: BLE001
                if i == attempts - 1:
                    raise
                time.sleep(wait)

    def _session(self, shape):
        options = self.ort.SessionOptions()
        if self.name == 'openvino':
            add_openvino_dll_dir()
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
            # OpenVINO 自己的編譯快取：Windows 的 NPU 驅動沒有磁碟快取，每次啟動 6 個形狀要重編約 20 s
            # （每個約 3.4 s），編譯完成前請求都走 CPU；有快取後約 5 s。Linux 的驅動另有 ze_intel_npu_cache。
            provider_options = {'device_type': 'NPU', 'cache_dir': accel_cache_dir('openvino')}
        else:
            length, proj_len = shape
            for dim, value in (('L', length), ('u', proj_len), ('k', NPU_ROWS), ('n', 1)):
                options.add_free_dimension_override_by_name(dim, value)
            provider_options = {'cache_dir': accel_cache_dir('vitisai'), 'cache_key': f'zc-{self.model_sha[:16]}-{length}x{proj_len}'}
            config = vitisai_config()
            if config:
                provider_options['config_file'] = config
        session = self.ort.InferenceSession(self.path, sess_options=options, providers=[(self.provider, provider_options)])
        # provider 建不起來（缺套件、沒有 NPU、沒有裝置權限）時 onnxruntime 只印警告並退回 CPU。
        if session.get_providers()[0] != self.provider:
            raise RuntimeError(f'{self.name}_npu_unavailable')
        return session

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
                session = self.shared or self._session(shape)
                with self.lock:
                    raw = session.run(['scores'], padded)[0]
                if not np.isfinite(raw).all():
                    raise ValueError('nonfinite_neural_score')
                self.sessions[shape] = session
                self.ready.add(shape)
            except Exception as e:   # noqa: BLE001 — 只停用這個形狀，CPU 照常
                self.unsupported.add(shape)
                print(f'npu ({self.name}) cannot compile shape {shape}, using cpu: {str(e).splitlines()[0][:200]}', file=sys.stderr, flush=True)
            finally:
                self.pending.discard(shape)

    def disable(self, error):
        if not self.failed:
            self.failed = True
            print(f'npu ({self.name}) disabled, using cpu: {error}', file=sys.stderr, flush=True)

    def run(self, inputs):
        """編譯過且 session 空著時回傳分數，否則回 None 讓呼叫端用 CPU。"""
        if self.failed:
            return None
        shape = npu_shape(inputs)
        if shape not in self.ready or not self.lock.acquire(blocking=False):
            return None
        try:
            raw = self.sessions[shape].run(['scores'], pad_for_npu(inputs, shape))[0]
        except Exception as e:   # noqa: BLE001
            self.disable(e)
            return None
        finally:
            self.lock.release()
        return unpad_scores(raw, inputs)


# >>> accel-shared：sequence_runtime.py 與 punct_runtime.py 逐字相同（worker 以 -I 執行，不能 import 同目錄的模組；
# internal/neural 的 TestAccelSharedBlockIdentical 核對）。
GPU_PROVIDERS = (
    # 依序試，第一個建得起來的就用。一律 FP32：遮罩以 float32 最小值相加，fp16 會溢位成 inf／NaN。
    # 不用 TensorRT：每個新形狀都要建引擎（數秒以上），遠超過宿主的期限。
    ('cuda', 'CUDAExecutionProvider', {'use_tf32': '0'}),   # NVIDIA；TF32 只有 10 位尾數，關掉以保持 FP32 結果
    ('migraphx', 'MIGraphXExecutionProvider', {}),          # AMD（ROCm 上的 MIGraphX；ROCm EP 已被官方標為 deprecated，不用）
    ('openvino', 'OpenVINOExecutionProvider', {'device_type': 'GPU', 'precision': 'FP32'}),   # Intel
)
GPU_MAX_ERRORS = 3
_openvino_plugin = None


def openvino_plugin(ort):
    """onnxruntime-ep-openvino（OpenVINO 的 plugin EP）的裝置：{'GPU': OrtEpDevice, 'NPU': …}；沒裝或載入失敗時是 {}。

    plugin 掛在一般的 onnxruntime（1.23 以上）上、wheel 不分 Python 版本、自帶 OpenVINO，所以內附的 Python 與
    onnxruntime 不必換；舊的 onnxruntime-openvino 要取代整個 onnxruntime，且只到 cp313、要配對 openvino 套件的版本。
    兩者都裝時用舊的（provider 已在 get_available_providers 裡）。plugin 一個行程只註冊一次。
    """
    global _openvino_plugin
    if _openvino_plugin is None:
        _openvino_plugin = {}
        try:
            import onnxruntime_ep_openvino as plugin
            ort.register_execution_provider_library('zc_openvino', plugin.get_library_path())
            name = plugin.get_ep_name()
            for device in ort.get_ep_devices():
                kind = str(device.ep_metadata.get('ov_device', '')).split('.')[0]   # GPU、GPU.0、NPU…
                if device.ep_name == name and kind in ('GPU', 'NPU'):
                    _openvino_plugin.setdefault(kind, device)
        except Exception:   # noqa: BLE001 — 沒裝 plugin、onnxruntime 太舊（沒有 plugin API）時就是沒有
            _openvino_plugin = {}
    return _openvino_plugin


def openvino_plugin_options(options, device, config):
    """把 plugin 裝置加進 SessionOptions；config 是 OpenVINO 的屬性（plugin 忽略 precision、cache_dir 等 provider 選項，
    要以 load_config 給）。OpenVINO 編譯失敗時會靜默改用自己的 CPU，所以一律關掉 CPU 退路。"""
    options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    name = str(device.ep_metadata.get('ov_device', ''))
    options.add_provider_for_devices([device], {'load_config': json.dumps({name: config})})


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


def open_gpu(ort, path):
    """依 GPU_PROVIDERS 的順序開第一個可用的 GPU session，回傳 (session, 名稱)；都不行時丟例外。

    一台電腦只裝得了一種 onnxruntime（cuda、migraphx、openvino 版互相取代），可用的 provider 由它決定；Intel 也可以是
    一般 onnxruntime 加上 OpenVINO 的 plugin（openvino_plugin）。
    provider 建不起來時 onnxruntime 只印警告並改用 CPU，所以以 get_providers() 確認真的是那個 provider。
    """
    available = set(ort.get_available_providers())
    errors = []
    for name, provider, provider_options in GPU_PROVIDERS:
        options = ort.SessionOptions()
        providers = [(provider, provider_options)]
        if provider not in available:
            device = openvino_plugin(ort).get('GPU') if provider == 'OpenVINOExecutionProvider' else None
            if device is None:
                continue
            # OpenVINO 的 GPU 預設以 f16 計算，要明確指定 f32。
            openvino_plugin_options(options, device, {'INFERENCE_PRECISION_HINT': 'f32'})
            providers = None
        elif provider == 'OpenVINOExecutionProvider':
            add_openvino_dll_dir()
            # OpenVINO 編譯失敗時會靜默改用自己的 CPU（比較慢），要它丟錯。其他 provider 本來就會把少數
            # 形狀運算分給 CPU EP，設了反而整個 session 開不起來，所以只對 OpenVINO 設。
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        try:
            session = ort.InferenceSession(path, sess_options=options, providers=providers)
        except Exception as e:   # noqa: BLE001 — 試下一個
            errors.append(f'{name}: {str(e).splitlines()[0][:120]}')
            continue
        if session.get_providers()[0] == provider:
            return session, name
        errors.append(f'{name}: fell back to {session.get_providers()[0]}')
    raise RuntimeError('no_gpu_provider' + (' (' + '; '.join(errors) + ')' if errors else ''))


class Offload:
    """在常駐的背景執行緒跑 GPU session；前景最多等 budget 秒，沒等到就這一次改用 CPU（回 None）。

    沒等到的那次照樣在背景跑完，等於替這個形狀暖機（有些 provider 每個新形狀第一次要編譯，久的要數秒）；
    跑完之前 session 占著，其他請求也走 CPU，請求永遠不等 GPU 超過 budget，宿主的期限也就不會因 GPU 逾時。
    session 只在同一個常駐執行緒上跑（OpenVINO 等 provider 依執行緒保留推論狀態，每次換執行緒會變慢）。
    執行出錯或輸出非有限值時這一次改用 CPU，連續 GPU_MAX_ERRORS 次即停用。
    """

    def __init__(self, session, name, output, budget):
        self.session, self.name, self.output, self.budget = session, name, output, budget
        self.idle, self.errors = True, 0
        self.jobs = queue.Queue(maxsize=1)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            inputs, box, done = self.jobs.get()
            try:
                box['out'] = self.session.run([self.output], inputs)[0]
            except Exception as e:   # noqa: BLE001
                box['error'] = e
            finally:
                self.idle = True
                done.set()

    def run(self, inputs):
        if self.errors >= GPU_MAX_ERRORS or not self.idle:
            return None
        box, done = {}, threading.Event()
        self.idle = False   # 只有前景（主執行緒）設成 False、只有背景設回 True
        self.jobs.put((inputs, box, done))
        if not done.wait(self.budget):
            return None
        out = box.get('out')
        if out is None or not np.isfinite(out).all():
            self.errors += 1
            if self.errors == GPU_MAX_ERRORS:
                reason = str(box.get('error', 'nonfinite output')).splitlines()[0][:200]
                print(f'gpu ({self.name}) disabled, using cpu: {reason}', file=sys.stderr, flush=True)
            return None
        self.errors = 0
        return out
# <<< accel-shared


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
        self.accel = None   # 加速裝置（NPU 或 Offload 包著的 GPU）；None 時只用 CPU
        if accel != 'cpu':
            try:
                if accel == 'npu':
                    self.accel = NPU(ort, str(package / 'model.onnx'), m['files']['model.onnx'])
                else:
                    self.accel = Offload(*open_gpu(ort, str(package / 'model.onnx')), 'scores', GPU_BUDGET)
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


# ---- 硬體加速偵測（--probe）：設定程式據此標出這台能用哪些選項 ----
# 只看兩件事：作業系統列出的裝置（Linux 的 /sys、Windows 登錄的 Display 與 ComputeAccelerator 裝置類別），以及已安裝的
# onnxruntime 有哪些 provider。「可用」不保證開得起來（驅動、plugin 有問題時仍會失敗），實際使用的裝置看就緒訊息的 accel。
PCI_VENDORS = {'0x10de': 'nvidia', '0x1002': 'amd', '0x1022': 'amd', '0x8086': 'intel', '0x5143': 'qualcomm'}
# provider 能用在哪些廠牌的裝置上。
PROVIDER_VENDORS = {'cuda': {'nvidia'}, 'migraphx': {'amd'}, 'openvino': {'intel'}, 'vitisai': {'amd'}}
NPU_DRIVERS = {'intel_vpu': ('intel', 'Intel NPU'), 'amdxdna': ('amd', 'AMD NPU')}
VENDOR_NAMES = {'nvidia': 'NVIDIA', 'amd': 'AMD', 'intel': 'Intel', 'qualcomm': 'Qualcomm'}


def vendor_of(text):
    t = text.lower()
    for key, vendor in (('nvidia', 'nvidia'), ('advanced micro', 'amd'), ('amd', 'amd'), ('radeon', 'amd'),
                        ('intel', 'intel'), ('qualcomm', 'qualcomm')):
        if key in t:
            return vendor
    return 'other'


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ''


def linux_devices(sys_root='/sys'):
    """PCI 顯示控制器（class 0x03xxxx）與已載入驅動的 NPU（/sys/class/accel）。"""
    gpus, npus = [], []
    pci = Path(sys_root, 'bus', 'pci', 'devices')
    for dev in sorted(pci.iterdir()) if pci.is_dir() else []:
        if _read(dev / 'class').startswith('0x03'):
            vendor = PCI_VENDORS.get(_read(dev / 'vendor'), 'other')
            driver = os.path.basename(os.readlink(dev / 'driver')) if (dev / 'driver').is_symlink() else ''
            gpus.append({'vendor': vendor, 'name': f'{VENDOR_NAMES.get(vendor, "Other")} GPU' + (f' ({driver})' if driver else '')})
    accel = Path(sys_root, 'class', 'accel')
    for node in sorted(accel.iterdir()) if accel.is_dir() else []:
        dev = node / 'device'
        driver = os.path.basename(os.readlink(dev / 'driver')) if (dev / 'driver').is_symlink() else ''
        vendor, name = NPU_DRIVERS.get(driver, (PCI_VENDORS.get(_read(dev / 'vendor'), 'other'), 'NPU'))
        npus.append({'vendor': vendor, 'name': name})
    return gpus, npus


def windows_devices():
    """登錄的裝置類別 Display（顯示卡）與 ComputeAccelerator（Windows「神經處理器」，NPU）底下已安裝驅動的裝置。"""
    import winreg
    gpus, npus = [], []
    root = r'SYSTEM\CurrentControlSet\Control\Class'

    def subkeys(key):
        i = 0
        while True:
            try:
                yield winreg.EnumKey(key, i)
            except OSError:
                return
            i += 1

    def value(key, name):
        try:
            return str(winreg.QueryValueEx(key, name)[0])
        except OSError:
            return ''

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root) as classes:
        for guid in subkeys(classes):
            try:
                with winreg.OpenKey(classes, guid) as cls:
                    kind = value(cls, 'Class')
                    if kind not in ('Display', 'ComputeAccelerator'):
                        continue
                    for sub in subkeys(cls):
                        if not sub.isdigit():
                            continue
                        try:
                            with winreg.OpenKey(cls, sub) as dev:
                                name, provider = value(dev, 'DriverDesc'), value(dev, 'ProviderName')
                        except OSError:
                            continue
                        if not name or (kind == 'Display' and provider.lower().startswith('microsoft')):
                            continue   # 基本顯示卡、遠端桌面等 Microsoft 的虛擬顯示卡不算
                        (gpus if kind == 'Display' else npus).append({'vendor': vendor_of(provider + ' ' + name), 'name': name})
            except OSError:
                continue
    return gpus, npus


def probe(sys_root='/sys'):
    """回報這台的 GPU／NPU 與 onnxruntime 的 provider，以及設定的 gpu、npu 各自能不能用。"""
    out = {'probe': 1}
    plugin = {}
    try:
        import onnxruntime as ort
        providers = list(ort.get_available_providers())
        out['onnxruntime'] = ort.__version__
        plugin = openvino_plugin(ort)
    except Exception as e:   # noqa: BLE001 — 沒有 onnxruntime 時裝置照樣列出
        providers = []
        out['onnxruntime_error'] = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
    out['providers'] = providers
    if plugin:
        out['openvino_plugin'] = sorted(plugin)
    try:
        gpus, npus = windows_devices() if sys.platform == 'win32' else linux_devices(sys_root)
    except Exception as e:   # noqa: BLE001
        gpus, npus = [], []
        out['devices_error'] = str(e)[:200]
    for kind, devices, table in (('gpu', gpus, GPU_PROVIDERS), ('npu', npus, NPU_PROVIDERS)):
        vendors = {d['vendor'] for d in devices}
        usable = [entry[0] for entry in table
                  if (entry[1] in providers or kind == 'gpu' and entry[0] == 'openvino' and 'GPU' in plugin)
                  and PROVIDER_VENDORS[entry[0]] & vendors]
        status = 'available' if usable and devices else 'no_device' if not devices else 'no_runtime'
        out[kind] = {'status': status, 'devices': devices, 'providers': usable}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path); p.add_argument('--dictionary')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--accel', choices=ACCELS, default='cpu',
                   help='gpu：第一個可用的 GPU provider（CUDA、MIGraphX、OpenVINO，FP32）；'
                        'npu：已編譯形狀走 NPU（OpenVINO 的 Intel NPU、VitisAI 的 AMD NPU）；裝置無法使用時走 CPU；'
                        '--probe 只偵測裝置')
    p.add_argument('--probe', action='store_true', help='只回報這台的 GPU／NPU 與可用的 provider（JSON 一行）後結束，不載入模型')
    p.add_argument('--probe-sys-root', default='/sys', help=argparse.SUPPRESS)   # 測試用的假 /sys
    a = p.parse_args()
    if a.probe:
        print(json.dumps(probe(a.probe_sys_root), ensure_ascii=False), flush=True)
        return
    if a.package is None or a.dictionary is None:
        p.error('--package and --dictionary are required')
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
    # 實際使用的裝置與 provider：要求 gpu／npu 但裝置無法使用時是 cpu（宿主的 health 回報 selection_accel、
    # selection_accel_provider）。
    ready['accel'] = 'npu' if isinstance(runtime.accel, NPU) else 'gpu' if isinstance(runtime.accel, Offload) else 'cpu'
    if runtime.accel is not None:
        ready['accel_provider'] = runtime.accel.name
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
