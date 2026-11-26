# 神經模型 runtime

完整桌面套件於系統字典旁提供 `neural/`，由 `0c-server` 依設定啟動 worker。神經選字預設開啟；補標點為 Beta、預設關閉。

`runtime/sequence_runtime.py` 與 `runtime/punct_runtime.py` 是 stdio worker，依賴 numpy 與 onnxruntime。固定模型版本、授權及系統字典配對要求見 [模型說明](../../models/README.md)。

`qsel/` 是手機選字模型（q4x-b16）不經 onnxruntime 的 C 推論，權重留在唯讀 mmap（#336，見 [qsel/README.md](qsel/README.md)）；桌面不用。

## 使用發行版提供的 Python

從完整 tarball 根目錄啟動已編譯的 server：

```sh
build/0c-server -data assets/data \
  -neural-python /usr/bin/python3 \
  -neural-package models/select -neural-worker platforms/neural/runtime/sequence_runtime.py \
  -punct-package models/punct -punct-worker platforms/neural/runtime/punct_runtime.py
```

系統 Python 須已安裝 numpy 與 onnxruntime；server 依使用者設定啟動模型。補標點仍預設關閉。安裝時依發行版目錄配置調整上述路徑，並保留字典與模型的固定配對。

### Intel GPU／NPU（選用）

設定的「硬體加速」（`neural-accel`，`settings.json` 的 `neural_accel`：`cpu`、`gpu`、`npu`，只有 Windows、Linux）讓選字與補標點 worker 改用 OpenVINO 的 Intel GPU 或 NPU（worker 參數 `--accel`）；預設 `cpu` 與以前相同，裝置無法使用時也照舊走 CPU，就緒訊息的 `accel` 回報實際使用的裝置（health 的 `selection_accel`，設定程式據此說明）。`0c-server -neural-accel` 可蓋過設定。兩者都需要 onnxruntime 含 OpenVINO EP（Arch：`python-onnxruntime-openvino`）；官網安裝包內附的 onnxruntime 只有 CPU，選 GPU、NPU 時照舊用 CPU。

在 Core Ultra 200V（Arc 140V）以逐鍵輸入的 3,472 個請求（前文 64 字）量測，往返延遲：

| 裝置 | 平均 | p99 | 首選與 CPU 相同 |
|---|---:|---:|---:|
| `cpu`（8 threads） | 11.2 ms | 19.0 ms | — |
| `gpu` | 4.6 ms | 6.6 ms | 100% |
| `npu` | 約 6.7 ms | 約 19 ms | 99.8% |

**GPU**（建議）需要 OpenVINO 的 GPU plugin 與 Intel 的 OpenCL 驅動（Arch：`openvino-intel-gpu-plugin`、`intel-compute-runtime`）。以 FP32 計算、收動態形狀，不必補齊；新形狀第一次約數十 ms。FP16 不能用：遮罩以 float32 最小值相加，fp16 溢位成 inf／NaN。

**補標點**只支援 GPU（FP32，`punctuation_accel` 回報實際裝置）：模型沒有遮罩輸入，補齊會改變結果，不能用 NPU 要的固定形狀，所以選 NPU 時補標點照常用 CPU。以 300 段 8–200 字的無標點文字量測，往返延遲平均 12.2 → 5.5 ms、p99 44 → 16 ms，補出的標點與 CPU 全部相同；新長度第一次最多約 80 ms，在補標點期限 150 ms 內。

**NPU** 需要 OpenVINO 的 NPU plugin（`openvino-intel-npu-plugin`）與 NPU 使用者空間驅動（`intel-npu-driver`、`level-zero-loader`），且使用者打得開 `/dev/accel/accel*`。

NPU 只吃固定形狀，worker 把輸入補齊到 6 個形狀（L 128／256 × proj_pos 8／16／24，target 256 列），就緒後在背景編譯；編譯好之前、超出這些形狀（逐鍵輸入約 22%，主要是 proj_pos 超過 24）、或 NPU 無法使用時，照舊用 CPU session，請求不等編譯。NPU 以 fp16 計算，模型分數與 CPU 略有差異（逐鍵請求的首選 99.8% 相同）。編譯結果快取在 `~/.cache/ze_intel_npu_cache`。

## 官網可攜套件

```sh
python3 scripts/prepare_neural.py --platform linux-amd64 \
  --dictionary assets/data --models models \
  --out build/neural-linux-amd64
```

Windows 使用 `windows-amd64`，Apple Silicon 使用 `macos-arm64`（onnxruntime 需 macOS 14 以上）。Intel Mac 使用 `macos-amd64`：onnxruntime 1.30.0 沒有 Intel Mac 版，改用最後一版有的 1.23.2 與它支援的 CPython 3.13.15；兩個模型的輸出與 1.30.0 逐請求比對過相同。工具下載並校驗固定版本的 Python 與 wheel，快取在 `build/cache/neural`。

`windows-amd64` 的 `neural/` 要在 Linux x86-64 上準備，再把輸出目錄複製到 Windows 建置機交給 `scripts/package_windows.py --neural`：工具用同一版 CPython 的 Linux 版（python-build-standalone 的 `x86_64-unknown-linux-gnu`）把 numpy、onnxruntime 編成 bytecode，這個直譯器在 Windows 上跑不起來。同一版 CPython 的 bytecode 不分平台，Windows 內附 Python 的標準函式庫本來就是 `python314.zip` 裡的 bytecode，所以不必在 Windows 上編。macOS 的兩個平台在 Linux 或 Mac 上準備都可以。

模型是 `models/models.lock.json` 記的 Hugging Face 公開版（#180，0.2.3 起；0.2.2 以前是權重相同的內部匯出，見 `models/equivalent.json`）。`--models` 指定的目錄須有 `<select|punct>/model.onnx`、`LICENSE`、`README.md`，完整 tarball 的 `models/` 即可；省略時用 `build/cache/models`，缺的從 Hugging Face 下載，一律逐檔核對 `models.lock`。`manifest.json`、`vocab.json` 由 `models/upstream` 配對到 `--dictionary`，結果等於這份字典的 `pairing.lock.json`。只換字典或只換模型、沿用現有的可攜 Python 時用 `scripts/hf_models.py neural`。

以 `--neural` 將輸出目錄交給平台打包工具，並用同一份字典作為 `--data`。發行版離線建置應自行提供相依套件，不直接使用會下載 runtime 的可攜套件流程。

worker 必須以 `-I -B -X utf8` 執行；安裝後檔案不得在執行時改變。換模型須同步更換配對字典並重新產生 runtime，不手動改寫 manifest。公開包打包時核對 `neural/` 就是公開版模型配對出貨字典的結果（`hf_models.check_shipped`）。
