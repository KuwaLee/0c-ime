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

### 硬體加速（選用）

設定的「硬體加速」（`neural-accel`，`settings.json` 的 `neural_accel`：`cpu`、`gpu`、`npu`，只有 Windows、Linux）讓選字與補標點 worker 改用顯示卡或 Intel NPU（worker 參數 `--accel`）；預設 `cpu` 與以前相同，裝置無法使用時也照舊走 CPU。就緒訊息的 `accel`、`accel_provider` 回報實際使用的裝置與 provider（health 的 `selection_accel`、`selection_accel_provider`、`punctuation_accel`、`punctuation_accel_provider`），設定程式與 `0c-update status` 據此說明。`0c-server -neural-accel` 可蓋過設定。官網安裝包內附的 onnxruntime 只有 CPU，選 GPU、NPU 時照舊用 CPU。

**GPU** 用 onnxruntime 裡第一個可用的 provider，依序試（`GPU_PROVIDERS`，兩個 worker 的 `accel-shared` 區塊逐字相同）：

| provider | 顯示卡 | 需要的 onnxruntime |
|---|---|---|
| `cuda`（CUDA EP，關掉 TF32） | NVIDIA | Arch `python-onnxruntime-cuda`；pip `onnxruntime-gpu` |
| `migraphx`、`rocm` | AMD（ROCm 支援的型號） | Arch `python-onnxruntime-rocm`；pip `onnxruntime-rocm` |
| `directml` | Windows 上任何 DirectX 12 顯示卡 | pip `onnxruntime-directml` |
| `openvino`（`device_type` GPU） | Intel | Arch 自行打包的 `python-onnxruntime-openvino` 加 `openvino-intel-gpu-plugin`、`intel-compute-runtime`；pip `onnxruntime-openvino`（Windows 另要 `openvino`） |

一台電腦只裝得了一種 onnxruntime（這些套件互相取代），可用的 provider 由它決定。一律以 FP32 計算、收動態形狀：遮罩以 float32 最小值相加，fp16 會溢位成 inf／NaN；不用 TensorRT，因為每個新形狀都要建引擎（數秒以上）。GPU 在常駐的背景執行緒上跑（`Offload`），每個請求最多等 30 ms（補標點每一窗 50 ms），沒等到就這一次用 CPU、GPU 在背景跑完當作暖機，所以新形狀編譯很久的 provider 也不會讓引擎逾時。

**補標點**只支援 GPU：模型沒有遮罩輸入，補齊會改變結果，不能用 NPU 要的固定形狀，所以選 NPU 時補標點照常用 CPU。

**NPU** 只有 Intel（OpenVINO 的 `device_type` NPU），需要 OpenVINO 的 NPU plugin（`openvino-intel-npu-plugin`）與 NPU 使用者空間驅動（`intel-npu-driver`、`level-zero-loader`），且使用者打得開 `/dev/accel/accel*`。NPU 只吃固定形狀，worker 把輸入補齊到 6 個形狀（L 128／256 × proj_pos 8／16／24，target 256 列），就緒後在背景編譯；編譯好之前、超出這些形狀（逐鍵輸入約 22%，主要是 proj_pos 超過 24：每個位置要對約 2.8 萬字的字表做 log_softmax，放不進 NPU 的片上記憶體）、或 NPU 無法使用時，照舊用 CPU，請求不等編譯。NPU 以 fp16 計算，分數與 CPU 略有差異。編譯結果快取在 `~/.cache/ze_intel_npu_cache`。AMD Ryzen AI 的 NPU（VitisAI EP）要先把模型量化、編譯成它的格式，模型檔會變，不在這裡支援。

在 Core Ultra 200V（Arc 140V，`openvino`）量測，往返延遲：

| | 請求 | CPU（8 threads）平均／p99 | 加速 平均／p99 | 與 CPU 相同 |
|---|---|---:|---:|---:|
| 選字 `gpu` | 逐鍵輸入 3,472 個（前文 64 字） | 11.2／19.0 ms | 4.6／6.6 ms | 首選 100% |
| 選字 `npu` | 同上 | 11.2／19.0 ms | 約 6.7／19 ms | 首選 99.8% |
| 補標點 `gpu` | 300 段 8–200 字 | 12.2／44 ms | 5.5／16 ms | 100% |

NPU 在這台的優點是不占 CPU、較省電，不是比較快：請求小的時候（例如沒有前文），8 threads 的 CPU 平均就和 NPU 差不多。

#### 必要套件

三樣都要齊：**驅動**（作業系統認得裝置）、**執行階段**（OpenCL、CUDA、ROCm 等）、**支援該 provider 的 onnxruntime**（取代原本只有 CPU 的那一個，與 worker 用的 Python 版本相符）。少任何一樣，worker 都照舊用 CPU，原因印在 stderr，`accel_check.py` 會列出來。

**Linux（Arch）**：系統 Python 用 `python-onnxruntime-*` 其中一個，它取代 `python-onnxruntime`。除了 `python-onnxruntime-openvino`，下表的套件都在 `extra`。

| 選項 | 裝置 | 必要套件 | 另外要做 |
|---|---|---|---|
| `gpu` | Intel 內顯 | `python-onnxruntime-openvino`（官方軟體庫沒有，要自行打包：onnxruntime 以 `--use_openvino` 建置）、`openvino-intel-gpu-plugin`、`intel-compute-runtime` | — |
| `gpu` | NVIDIA | `python-onnxruntime-cuda`（依賴帶進 `cuda`、`cudnn`）、`nvidia` 或 `nvidia-open` 驅動 | — |
| `gpu` | AMD | `python-onnxruntime-rocm`（依賴帶進 ROCm 執行階段）；顯示卡要在 ROCm 支援清單內 | 使用者在 `render`、`video` 群組 |
| `npu` | Intel NPU（Core Ultra） | `python-onnxruntime-openvino`、`openvino-intel-npu-plugin`、`intel-npu-driver`、`level-zero-loader` | 使用者打得開 `/dev/accel/accel*`（見下） |

`intel-npu-driver` 的 udev 規則（`10-intel-npu.rules`）把 `/dev/accel/accel*` 設成 `root:render 0660`，systemd 預設的 `50-udev-default.rules` 再設回 `0666`；剛裝好驅動時可能停在 0660，`sudo udevadm trigger --subsystem-match=accel --action=add` 或重新開機後恢復，或把使用者加進 `render` 群組（重新登入生效）。

**Linux（Debian、Ubuntu 等）**：發行版沒有 GPU 版的 `python3-onnxruntime`。在 venv 以 pip 裝下表 Windows 那一欄的套件（NVIDIA 用 `onnxruntime-gpu`、AMD 用 `onnxruntime-rocm`、Intel 用 `onnxruntime-openvino`），驅動與執行階段照各廠商的說明裝，再以 `0c-server -neural-python <venv>/bin/python3` 指向它。

**Windows**：官網安裝包內附的 Python 3.14 與 onnxruntime 只有 CPU，要另外準備一個 Python（`onnxruntime-openvino` 只有到 Python 3.13 的 wheel，建議 3.13），裝下表的套件，再讓 `0c-server` 以 `-neural-python` 指向它（安裝包本身還沒內附加速元件）。

| 選項 | 裝置 | pip 套件 | 驅動與執行階段 |
|---|---|---|---|
| `gpu` | 任何 DirectX 12 顯示卡（NVIDIA、AMD、Intel） | `numpy onnxruntime-directml` | 顯示卡驅動（DirectML 隨套件附） |
| `gpu` | NVIDIA（CUDA） | `numpy onnxruntime-gpu` | NVIDIA 驅動，以及 onnxruntime 版本要求的 CUDA 12 與 cuDNN 9 |
| `gpu`、`npu` | Intel 內顯、Intel NPU | `numpy onnxruntime-openvino==1.24.1 openvino` | Intel 顯示驅動；NPU 另要 Intel NPU 驅動 |

`onnxruntime-openvino` 在 Windows 需要 `openvino` 套件的 DLL，worker 會自己把 `openvino\libs` 加進 DLL 搜尋路徑。這些套件互相取代，一個 Python 環境只裝一種。

#### 驗證（`accel_check.py`）

其他顯示卡沒有在開發機上量過。用安裝好的模型與要給 worker 用的 Python（含 numpy 與支援加速的 onnxruntime）執行，CPU 與加速各跑一次同一批請求，比較實際裝置、與 CPU 是否相同、延遲與超過引擎期限的次數，最後印 PASS／SLOWER／DIFFERS／FAIL（只有 PASS 時結束碼是 0）：

```sh
python3 platforms/neural/accel_check.py --select <模型目錄>/select --punct <模型目錄>/punct --accel gpu --json accel-report.json
```

請求在本機依模型字表產生，不含使用者文字、不連網；`--requests` 可改用錄下的選字請求（JSONL），`--texts` 改用補標點文字。產生的候選是隨機字，分數彼此接近，NPU（fp16）比真實輸入更容易排錯，這時差異超過門檻記為 DIFFERS 而不是 FAIL。`--json` 只存統計數字與環境（作業系統、Python、onnxruntime 版本與可用 provider），可附在回報裡。

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
