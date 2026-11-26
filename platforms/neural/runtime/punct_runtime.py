#!/usr/bin/env python3
"""提交時補標點的 ONNX worker（zuin-punct-runtime-v1）。

協定（每行一個 JSON）：啟動後先輸出 ready；之後每行請求 {"id", "text"}，回覆 {"id", "edits": [{"offset", "text"}]}
或 {"id", "error"}。只插入標點、不改原字；offset 是原字串的字元（rune）索引，插在該字元之前。
插入規則：可插入位置由 eligible_boundaries 決定，段尾不補；各位置取標點機率最高者，
≥ 套件門檻才插入；開頭單個數字後不補頓號（見 ENUMERATORS）。長於一個窗口時依 manifest 的窗口與步長滑動，
每個位置取離窗口邊緣最遠的那一窗。manifest 的 input 決定模型輸入：echo（回聲輸入，窗口 64、步長 32；
舊套件沒有此欄位時的預設）或 bidirectional（punct-encoder-20260930 雙向 encoder，[BOS][任務] 原文，窗口 128、步長 64）。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import sys
import threading

import numpy as np

# onnxruntime 在 import 時（建立全域 Env）就啟動遙測：Linux／macOS 連到外部伺服器並在個人目錄寫裝置識別碼，
# Windows 寫 ETW 給系統的診斷資料服務。這兩個變數只在那時讀取，必須在 import onnxruntime 之前設定；
# 宿主也會設（internal/neural 的 NoTelemetryEnv，說明在那裡）。
os.environ['ORT_DISABLE_TELEMETRY'] = '1'
os.environ['ORT_RUNNING_UNIT_TESTS'] = '1'

SCHEMA = 'zuin-punct-runtime-v1'
LABELS = ['', '，', '。', '？', '！', '、', '；']
URL = re.compile(r'(?:https?://|www\.)[^\s，。！？、；「」『』（）【】]+', re.I)
MAX_TEXT = 256
# 新聞語料的「一、」「二、」條列標題讓模型在開頭數字後補頓號（「一本正經」→「一、本正經」）；輸入法提交的是片段，不補。
ENUMERATORS = set('一二三四五六七八九十壹貳參肆伍陸柒捌玖拾甲乙丙丁戊己庚辛壬癸')
MAX_FRAME = 1 << 16
# GPU 每一窗最多等這麼久（秒），沒等到就這一窗用 CPU：宿主期限 150 ms，長文要跑好幾窗。
GPU_BUDGET = 0.05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_han(char):
    return '㐀' <= char <= '鿿' or '\U00020000' <= char <= '\U000323af'


def eligible_boundaries(text):
    """與 punctuation_data.eligible_boundaries 逐字相同（worker 不依賴訓練程式）。"""
    protected = set()
    for match in URL.finditer(text):
        protected.update(range(match.start(), match.end()))
    allowed = []
    for i, char in enumerate(text):
        following = text[i + 1:]
        next_char = following[0] if following else ''
        next_visible = following.lstrip()[:1]
        allowed.append(is_han(char) and i not in protected
                       and (not next_char or is_han(next_char) or next_char.isspace())
                       and (not next_visible or is_han(next_visible)
                            or next_visible.isascii() and next_visible.isalnum()))
    return allowed


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
    """accel='gpu'：另開第一個可用的 GPU session（open_gpu，FP32）；開不起來時只用 CPU。

    這個模型沒有遮罩輸入，補齊會改變結果，所以不支援要求固定形狀的 NPU。
    """

    def __init__(self, package, threads=2, accel='cpu'):
        import onnxruntime as ort
        if hasattr(ort, 'disable_telemetry_events'):   # 再保險一次：之後的 session 與推論事件也不記
            ort.disable_telemetry_events()
        package = Path(package)
        self.manifest = m = json.loads((package / 'manifest.json').read_text())
        if m['schema'] != SCHEMA or m['labels'] != LABELS:
            raise ValueError('package_contract_mismatch')
        for name in ('model.onnx', 'vocab.json'):
            if sha(package / name) != m['files'][name]:
                raise ValueError('package_hash_mismatch')
        vocab = json.loads((package / 'vocab.json').read_text())
        self.chars = {c: i + 8 for i, c in enumerate(vocab['characters'])}
        self.threshold, self.window, self.stride = float(m['threshold']), int(m['window']), int(m['stride'])
        self.input = m.get('input', 'echo')
        limit = {'echo': 64, 'bidirectional': 128}.get(self.input)
        if limit is None or not 0 < self.threshold < 1 or not 8 <= self.window <= limit or not 0 < self.stride <= self.window:
            raise ValueError('package_contract_mismatch')
        options = ort.SessionOptions(); options.intra_op_num_threads = threads; options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(package / 'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        self.model_sha = m['files']['model.onnx']
        self.gpu = None   # Offload 包著的 GPU session；None 時只用 CPU
        if accel == 'gpu':
            try:
                self.gpu = Offload(*open_gpu(ort, str(package / 'model.onnx')), 'probs', GPU_BUDGET)
            except Exception as e:   # noqa: BLE001 — 裝置無法使用時照舊只用 CPU
                print(f'gpu unavailable, using cpu: {e}', file=sys.stderr, flush=True)

    def run(self, inputs):
        """GPU 在 budget 內答得出來就用 GPU，否則這一窗用 CPU。"""
        probs = self.gpu.run(inputs) if self.gpu is not None else None
        return probs if probs is not None else self.session.run(['probs'], inputs)[0]

    def window_probs(self, text):
        src = [self.chars.get(c, 4) for c in text]
        if self.input == 'bidirectional':
            ids = np.array([[1, 7] + src], dtype=np.int64)
            return self.run({'ids': ids})[2:]
        ids = np.array([[1, 7] + src + [2] + src], dtype=np.int64)
        types = np.array([[0] * (len(src) + 3) + [2] * len(src)], dtype=np.int64)
        probs = self.run({'ids': ids, 'types': types})
        return probs[len(src) + 3:]

    def probabilities(self, text):
        n, W, S = len(text), self.window, self.stride
        starts = [0] if n <= W else list(range(0, n - W, S)) + [n - W]
        out, best = [None] * n, [-1] * n
        for s in starts:
            probs = self.window_probs(text[s:s + W]); m = len(probs)
            for j in range(m):
                margin = min(j, m - 1 - j) if n > W else 0
                if margin > best[s + j]:
                    best[s + j] = margin; out[s + j] = probs[j]
        return np.stack(out)

    def edits(self, text):
        if not isinstance(text, str) or not text or len(text) > MAX_TEXT:
            raise ValueError('unsupported_text')
        allowed = eligible_boundaries(text)
        if not any(allowed[:-1]):
            return []
        probs = self.probabilities(text)
        out = []
        for i in range(len(text) - 1):   # 段尾不補
            k = int(np.argmax(probs[i, 1:])) + 1
            if i == 0 and LABELS[k] == '、' and text[0] in ENUMERATORS:
                continue
            if allowed[i] and float(probs[i, k]) >= self.threshold:
                out.append({'offset': i + 1, 'text': LABELS[k]})
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--accel', choices=('cpu', 'gpu'), default='cpu', help='gpu：第一個可用的 GPU provider（CUDA、MIGraphX、OpenVINO，FP32）；無法使用時走 CPU')
    a = p.parse_args()
    if not 1 <= a.threads <= 8:
        p.error('threads must be between 1 and 8')
    runtime = Runtime(a.package, a.threads, a.accel)
    runtime.edits('暖機只用固定字串不含使用者文字')
    # accel、accel_provider：實際使用的裝置與 provider（宿主的 health 回報 punctuation_accel、punctuation_accel_provider）。
    ready = {'ready': True, 'schema': SCHEMA, 'model_sha256': runtime.model_sha, 'accel': 'gpu' if runtime.gpu is not None else 'cpu'}
    if runtime.gpu is not None:
        ready['accel_provider'] = runtime.gpu.name
    print(json.dumps(ready), flush=True)
    while True:
        line = sys.stdin.buffer.readline(MAX_FRAME + 1)
        if not line:
            return
        if len(line) > MAX_FRAME or not line.endswith(b'\n'):
            raise ValueError('request_frame_limit')
        r = json.loads(line)
        reply = {'id': r['id']}
        try:
            reply['edits'] = runtime.edits(r['text'])
        except (ValueError, KeyError, TypeError) as e:
            reply['error'] = str(e) if isinstance(e, ValueError) else 'bad_request'
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + '\n'); sys.stdout.flush()


if __name__ == '__main__':
    main()
