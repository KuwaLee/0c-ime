package main

import (
	"fmt"
	"io"
	"runtime"
	"strings"

	"github.com/0c-biggo/0c-ime/internal/autoupdate"
)

// modelStatus 只查詢已執行的引擎，不為了查看狀態啟動引擎或模型。
func modelStatus(dial autoupdate.Dialer) map[string]any {
	info, running, err := autoupdate.Health(dial)
	if err != nil {
		return map[string]any{"state": "unavailable"}
	}
	if !running {
		return map[string]any{"state": "not_running"}
	}
	state := map[string]any{"state": "running", "version": info["version"]}
	for _, key := range []string{"selection_state", "punctuation_state"} {
		value := info[key]
		if value == "" {
			value = "unknown" // 舊引擎只有啟動配置，不能據此推論模型仍可用。
		}
		state[key] = value
	}
	for _, key := range []string{"selection_reason", "punctuation_reason", "selection_source", "punctuation_source", "selection_accel", "punctuation_accel", "selection_accel_provider", "punctuation_accel_provider",
		// 硬體加速偵測（neural.Probe）：設定程式據此標出這台能用哪些「硬體加速」選項。
		"accel_probe", "accel_gpu", "accel_gpu_devices", "accel_gpu_vendors", "accel_gpu_providers",
		"accel_npu", "accel_npu_devices", "accel_npu_vendors", "accel_npu_providers"} {
		if info[key] != "" {
			state[key] = info[key]
		}
	}
	// #169：引擎可以即時切換「輸出簡體字」時附上它目前的值；外殼的選單據此打勾（比設定檔新：剛從別處切換過）。
	// 舊引擎沒有這兩項，外殼看 settings.output_simplified。
	if info["output_simplified_live"] == "true" {
		state["output_simplified_live"] = true
		state["output_simplified"] = info["output_simplified"] == "true"
	}
	return state
}

func printModelStatus(w io.Writer, state map[string]any) {
	switch state["state"] {
	case "not_running":
		fmt.Fprintln(w, "引擎目前未執行；模型運作狀態尚無法確認。")
	case "unavailable":
		fmt.Fprintln(w, "暫時無法查詢模型運作狀態。")
	default:
		fmt.Fprintf(w, "模型運作狀態（引擎 %s）：\n神經選字：%s\n補標點：%s\n",
			state["version"], modelStateLabel(state["selection_state"], "統計引擎")+reasonLabel(state["selection_reason"])+sourceLabel(state["selection_source"])+accelLabel(state["selection_accel"], state["selection_accel_provider"]),
			modelStateLabel(state["punctuation_state"], "原文提交")+reasonLabel(state["punctuation_reason"])+sourceLabel(state["punctuation_source"])+accelLabel(state["punctuation_accel"], state["punctuation_accel_provider"]))
		if line := probeLine(state); line != "" {
			fmt.Fprintln(w, line)
		}
	}
}

// modelHintOS 決定 runtime_missing 的提示；測試換成別的平台。
var modelHintOS = runtime.GOOS

// runtimeMissingLabel：只有 Linux（系統安裝用系統的 python3）提示要裝的 Debian 套件；macOS、Windows 的
// Python 與推論元件內附在安裝包裡，不提 apt 的套件名（與設定程式、Windows 選單相同）。
func runtimeMissingLabel(goos, fallback string) string {
	if goos == "linux" {
		return "缺少推論元件（請安裝 python3-numpy、python3-onnxruntime），使用" + fallback
	}
	return "缺少推論元件，使用" + fallback
}

func modelStateLabel(value any, fallback string) string {
	switch value {
	case "ready":
		return "已就緒"
	case "loading":
		return "載入中，暫用" + fallback
	case "fallback":
		return "模型無法使用，已退回" + fallback
	case "disabled_by_setting":
		return "依設定關閉"
	case "not_installed":
		return "未附模型，使用" + fallback
	case "runtime_missing": // #181：找到模型，但沒有 python3，或它缺 numpy、onnxruntime
		return runtimeMissingLabel(modelHintOS, fallback)
	case "closed":
		return "已停止，使用" + fallback
	default:
		return "未知（引擎未提供即時狀態）"
	}
}

// sourceLabel 說明模型從哪裡載入（#181）；安裝包內建的不另外註明。
func sourceLabel(value any) string {
	if value == "user" {
		return "（個人目錄的模型）"
	}
	return ""
}

// reasonLabel 說明模型退回的原因（引擎的 *_reason 代碼）；沒有原因時為空字串。
func reasonLabel(value any) string {
	switch value {
	case "missing_runtime":
		return "（缺少系統元件：Visual C++ 執行階段）"
	case "timeout":
		return "（模型載入逾時）"
	case "failed":
		return "（模型程式無法啟動）"
	}
	return ""
}

// accelLabel 說明模型實際使用的硬體加速（health 的 *_accel 與 *_accel_provider）；CPU 或舊引擎沒有回報時不加字。
func accelLabel(value, provider any) string {
	name := providerNames[fmt.Sprint(provider)]
	switch value {
	case "gpu":
		if name == "" {
			return "（GPU 加速）"
		}
		return "（GPU 加速，" + name + "）"
	case "npu":
		if name == "" {
			return "（NPU 加速）"
		}
		return "（NPU 加速，" + name + "）"
	}
	return ""
}

// providerNames 是 provider 代碼的顯示名稱（與設定程式的 ACCEL_PROVIDER 相同）。
var providerNames = map[string]string{"cuda": "NVIDIA CUDA", "migraphx": "AMD MIGraphX", "rocm": "AMD ROCm","openvino": "Intel OpenVINO", "vitisai": "AMD Ryzen AI"}

// probeLine 列出硬體加速偵測的結果（health 的 accel_*）；引擎沒有偵測（舊版、沒有選字 worker）時為空字串。
func probeLine(state map[string]any) string {
	if state["accel_probe"] == "failed" {
		return "硬體加速偵測：無法偵測"
	}
	if state["accel_probe"] != "ok" {
		return ""
	}
	part := func(kind, label string) string {
		devices, _ := state["accel_"+kind+"_devices"].(string)
		providers, _ := state["accel_"+kind+"_providers"].(string)
		switch state["accel_"+kind] {
		case "available":
			names := []string{}
			for _, p := range strings.Split(providers, ",") {
				if n := providerNames[p]; n != "" {
					names = append(names, n)
				}
			}
			return label + " 可用（" + devices + "；" + strings.Join(names, "、") + "）"
		case "no_runtime":
			return label + " 有裝置（" + devices + "），推論元件不支援"
		}
		return label + " 未偵測到"
	}
	return "硬體加速偵測：" + part("gpu", "GPU") + "；" + part("npu", "NPU")
}
