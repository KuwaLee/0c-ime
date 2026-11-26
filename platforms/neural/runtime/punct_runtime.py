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
import re
import sys

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


def add_openvino_dll_dir():
    """Windows：onnxruntime-openvino 的 provider DLL 依賴 openvino 套件的 DLL（openvino\\libs），不在預設搜尋路徑。

    與 sequence_runtime.py 相同（worker 以 -I 執行，不能 import 同目錄的模組）：找不到就不做事，建立 session 失敗後照舊用 CPU。
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


class Runtime:
    """accel='gpu'：另開 OpenVINO Intel GPU session（FP32，與 CPU 的補標點結果相同）；開不起來時只用 CPU。

    這個模型沒有遮罩輸入，補齊會改變結果，所以不支援要求固定形狀的 NPU。GPU 執行出錯或機率非有限值時，
    這一窗改用 CPU；連續出錯 GPU_MAX_ERRORS 次即停用 GPU。
    """
    GPU_MAX_ERRORS = 3

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
        self.gpu, self.gpu_errors = None, 0
        if accel == 'gpu':
            try:
                add_openvino_dll_dir()
                gpu_options = ort.SessionOptions()
                gpu_options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
                gpu = ort.InferenceSession(str(package / 'model.onnx'), sess_options=gpu_options,
                                           providers=[('OpenVINOExecutionProvider', {'device_type': 'GPU', 'precision': 'FP32'})])
                if gpu.get_providers()[0] != 'OpenVINOExecutionProvider':
                    raise RuntimeError('openvino_gpu_unavailable')
                self.gpu = gpu
            except Exception as e:   # noqa: BLE001 — 裝置無法使用時照舊只用 CPU
                print(f'gpu unavailable, using cpu: {e}', file=sys.stderr, flush=True)

    def run(self, inputs):
        """GPU 可用時用 GPU，出錯或非有限值時這一次改用 CPU。"""
        if self.gpu is not None and self.gpu_errors < self.GPU_MAX_ERRORS:
            try:
                probs = self.gpu.run(['probs'], inputs)[0]
                if not np.isfinite(probs).all():
                    raise ValueError('nonfinite_probs')
                self.gpu_errors = 0
                return probs
            except Exception as e:   # noqa: BLE001
                self.gpu_errors += 1
                if self.gpu_errors == self.GPU_MAX_ERRORS:
                    print(f'gpu disabled, using cpu: {str(e).splitlines()[0][:200]}', file=sys.stderr, flush=True)
        return self.session.run(['probs'], inputs)[0]

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
    p.add_argument('--accel', choices=('cpu', 'gpu'), default='cpu', help='gpu：OpenVINO Intel GPU（FP32）；無法使用時走 CPU')
    a = p.parse_args()
    if not 1 <= a.threads <= 8:
        p.error('threads must be between 1 and 8')
    runtime = Runtime(a.package, a.threads, a.accel)
    runtime.edits('暖機只用固定字串不含使用者文字')
    # accel：實際使用的裝置（宿主的 health 回報 punctuation_accel）。
    print(json.dumps({'ready': True, 'schema': SCHEMA, 'model_sha256': runtime.model_sha,
                      'accel': 'gpu' if runtime.gpu is not None else 'cpu'}), flush=True)
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
