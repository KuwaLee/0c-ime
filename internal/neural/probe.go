package neural

import (
	"bytes"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"time"
	"unicode"
)

// 硬體加速偵測（sequence_runtime.py --probe）的結果代碼：設定的 gpu、npu 在這台能不能用。
const (
	AccelAvailable = "available"  // 有裝置，onnxruntime 也有對應的 provider
	AccelNoDevice  = "no_device"  // 作業系統沒有列出這類裝置
	AccelNoRuntime = "no_runtime" // 有裝置，但已安裝的 onnxruntime 沒有對應的 provider
)

// Vendors 是偵測回報的裝置廠牌代碼。
var Vendors = []string{"nvidia", "amd", "intel", "qualcomm", "other"}

// probeTimeout：只 import onnxruntime、讀 /sys 或登錄，平常 0.2 秒；慢的電腦第一次 import 也遠在這之內。
const probeTimeout = 20 * time.Second

// Probe 以 worker 的 --probe 偵測這台的 GPU／NPU 與 onnxruntime 的 provider（不載入模型），回傳給 health 的欄位：
//
//	accel_gpu、accel_npu                       AccelAvailable／AccelNoDevice／AccelNoRuntime
//	accel_gpu_devices、accel_npu_devices       裝置名稱，以「；」分隔
//	accel_gpu_vendors、accel_npu_vendors       廠牌代碼（Vendors），以逗號分隔
//	accel_gpu_providers、accel_npu_providers   可用的 provider（Providers），以逗號分隔
//
// 只轉交認得的代碼；裝置名稱去掉控制字元並限制長度。舊 worker 不認得 --probe 時回錯誤。
func Probe(python, script string) (map[string]string, error) {
	cmd := pythonCommand(python, script, "--probe") // 與 worker 相同的隔離與環境（TestWorkersOnlyStartThroughPythonCommand）
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			return nil, err
		}
	case <-time.After(probeTimeout):
		_ = cmd.Process.Kill()
		<-done
		return nil, ErrTimeout
	}
	return parseProbe(out.Bytes())
}

type probeDevice struct {
	Vendor string `json:"vendor"`
	Name   string `json:"name"`
}

type probeKind struct {
	Status    string        `json:"status"`
	Devices   []probeDevice `json:"devices"`
	Providers []string      `json:"providers"`
}

func parseProbe(out []byte) (map[string]string, error) {
	var p struct {
		Probe int       `json:"probe"`
		GPU   probeKind `json:"gpu"`
		NPU   probeKind `json:"npu"`
	}
	if err := json.Unmarshal(out, &p); err != nil {
		return nil, err
	}
	if p.Probe != 1 {
		return nil, errors.New("unsupported accel probe")
	}
	info := map[string]string{}
	for kind, k := range map[string]probeKind{"gpu": p.GPU, "npu": p.NPU} {
		if !slices.Contains([]string{AccelAvailable, AccelNoDevice, AccelNoRuntime}, k.Status) {
			return nil, errors.New("invalid accel probe status")
		}
		info["accel_"+kind] = k.Status
		var names, vendors, providers []string
		for _, d := range k.Devices {
			if name := cleanDeviceName(d.Name); name != "" && len(names) < 4 {
				names = append(names, name)
			}
			if slices.Contains(Vendors, d.Vendor) && !slices.Contains(vendors, d.Vendor) {
				vendors = append(vendors, d.Vendor)
			}
		}
		for _, pr := range k.Providers {
			if slices.Contains(Providers, pr) && !slices.Contains(providers, pr) {
				providers = append(providers, pr)
			}
		}
		set := func(key string, v []string, sep string) {
			if len(v) > 0 {
				info["accel_"+kind+"_"+key] = strings.Join(v, sep)
			}
		}
		set("devices", names, "；")
		set("vendors", vendors, ",")
		set("providers", providers, ",")
	}
	return info, nil
}

// cleanDeviceName：裝置名稱來自登錄或 /sys，給設定程式顯示；去掉控制字元、分隔用的「；」，最長 80 字。
func cleanDeviceName(s string) string {
	s = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) || r == '；' {
			return -1
		}
		return r
	}, s)
	s = strings.TrimSpace(s)
	if r := []rune(s); len(r) > 80 {
		s = string(r[:80])
	}
	return s
}
