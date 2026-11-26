package neural

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseProbe(t *testing.T) {
	out := []byte(`{"probe":1,"providers":["CUDAExecutionProvider","CPUExecutionProvider"],
		"gpu":{"status":"available","devices":[{"vendor":"nvidia","name":"NVIDIA GeForce RTX 4060\u0007"},{"vendor":"intel","name":"Intel Arc 140V"}],"providers":["cuda","tensorrt"]},
		"npu":{"status":"no_runtime","devices":[{"vendor":"amd","name":"AMD NPU；x"}],"providers":[]}}`)
	info, err := parseProbe(out)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"accel_gpu": "available", "accel_gpu_devices": "NVIDIA GeForce RTX 4060；Intel Arc 140V", "accel_gpu_vendors": "nvidia,intel",
		"accel_gpu_providers": "cuda", // tensorrt 不在 Providers，不轉交
		"accel_npu":           "no_runtime", "accel_npu_devices": "AMD NPUx", "accel_npu_vendors": "amd",
	}
	if len(info) != len(want) {
		t.Fatalf("欄位：%v", info)
	}
	for k, v := range want {
		if info[k] != v {
			t.Fatalf("%s=%q，要 %q（%v）", k, info[k], v, info)
		}
	}
	for _, bad := range []string{`{"probe":2,"gpu":{"status":"available"},"npu":{"status":"no_device"}}`,
		`{"probe":1,"gpu":{"status":"maybe"},"npu":{"status":"no_device"}}`, `not json`} {
		if _, err := parseProbe([]byte(bad)); err == nil {
			t.Fatalf("要拒絕：%s", bad)
		}
	}
}

// 以真的 worker 偵測假的 /sys：NVIDIA 顯示卡與 AMD NPU，onnxruntime 沒有對應的 provider 時是 no_runtime；什麼都沒有時是 no_device。
func TestProbeWorkerWithFakeSys(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil || exec.Command(python, "-c", "import numpy").Run() != nil {
		t.Skip("需要 python3 與 numpy")
	}
	script, _ := filepath.Abs(filepath.Join("..", "..", "platforms", "neural", "runtime", "sequence_runtime.py"))
	root := t.TempDir()
	write := func(path, body string) {
		os.MkdirAll(filepath.Dir(filepath.Join(root, path)), 0o755)
		os.WriteFile(filepath.Join(root, path), []byte(body), 0o644)
	}
	link := func(target, path string) {
		os.MkdirAll(filepath.Dir(filepath.Join(root, path)), 0o755)
		if err := os.Symlink(target, filepath.Join(root, path)); err != nil {
			t.Fatal(err)
		}
	}
	write("bus/pci/devices/0000:01:00.0/class", "0x030000\n")
	write("bus/pci/devices/0000:01:00.0/vendor", "0x10de\n")
	os.MkdirAll(filepath.Join(root, "bus/pci/drivers/nvidia"), 0o755)
	link("../../../bus/pci/drivers/nvidia", "bus/pci/devices/0000:01:00.0/driver")
	write("devices/npu/vendor", "0x1022\n")
	os.MkdirAll(filepath.Join(root, "bus/pci/drivers/amdxdna"), 0o755)
	link("../../bus/pci/drivers/amdxdna", "devices/npu/driver")
	link("../../../devices/npu", "class/accel/accel0/device")
	empty := t.TempDir()

	run := func(sysRoot string) map[string]string {
		cmd := pythonCommand(python, script, "--probe", "--probe-sys-root", sysRoot)
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("--probe：%v", err)
		}
		info, err := parseProbe(out)
		if err != nil {
			t.Fatalf("%v：%s", err, out)
		}
		return info
	}
	info := run(root)
	if info["accel_gpu_vendors"] != "nvidia" || !strings.HasPrefix(info["accel_gpu_devices"], "NVIDIA GPU") ||
		info["accel_npu_vendors"] != "amd" || info["accel_npu_devices"] != "AMD NPU" {
		t.Fatalf("裝置：%v", info)
	}
	// 這台的 onnxruntime 不論有沒有 CUDA、VitisAI，結果都要一致：有 provider 才 available，沒有就是 no_runtime。
	for _, kind := range []string{"gpu", "npu"} {
		if s := info["accel_"+kind]; (s == AccelAvailable) != (info["accel_"+kind+"_providers"] != "") || s == AccelNoDevice {
			t.Fatalf("%s：%v", kind, info)
		}
	}
	if none := run(empty); none["accel_gpu"] != AccelNoDevice || none["accel_npu"] != AccelNoDevice {
		t.Fatalf("沒有裝置：%v", none)
	}
}
