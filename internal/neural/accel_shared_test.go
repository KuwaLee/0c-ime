package neural

import (
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// 兩個 worker 以 -I 執行、不能 import 同目錄的模組，所以 GPU 的 provider 選擇與背景執行（accel-shared 區塊）
// 各存一份；改了一邊忘了另一邊時這裡會失敗。
func TestAccelSharedBlockIdentical(t *testing.T) {
	block := func(name string) string {
		b, err := os.ReadFile(filepath.Join("..", "..", "platforms", "neural", "runtime", name))
		if err != nil {
			t.Fatal(err)
		}
		s := string(b)
		start, end := strings.Index(s, "# >>> accel-shared"), strings.Index(s, "# <<< accel-shared")
		if start < 0 || end < start {
			t.Fatalf("%s 沒有 accel-shared 區塊", name)
		}
		return s[start:end]
	}
	seq, punct := block("sequence_runtime.py"), block("punct_runtime.py")
	if seq != punct {
		t.Fatal("sequence_runtime.py 與 punct_runtime.py 的 accel-shared 區塊不同")
	}
	// Go 端認得的 provider 要與 worker 一致：GPU_PROVIDERS 依序，再加上 NPU_PROVIDERS 裡 GPU 沒有的（vitisai）。
	re := regexp.MustCompile(`\('([a-z]+)', '[A-Za-z]+ExecutionProvider'`)
	var names []string
	for _, m := range re.FindAllStringSubmatch(seq, -1) {
		names = append(names, m[1])
	}
	whole, err := os.ReadFile(filepath.Join("..", "..", "platforms", "neural", "runtime", "sequence_runtime.py"))
	if err != nil {
		t.Fatal(err)
	}
	npu := string(whole)
	npu = npu[strings.Index(npu, "NPU_PROVIDERS = ("):]
	npu = npu[:strings.Index(npu, "\n)\n")]
	for _, m := range re.FindAllStringSubmatch(npu, -1) {
		if !slices.Contains(names, m[1]) {
			names = append(names, m[1])
		}
	}
	if !slices.Equal(names, Providers) {
		t.Fatalf("worker 的 provider %v 與 Providers %v 不同", names, Providers)
	}
	if !strings.Contains(seq, "'use_tf32': '0'") {
		t.Fatal("CUDA 要關掉 TF32，結果才與 CPU 的 FP32 相同")
	}
}

func TestReadyProvider(t *testing.T) {
	for _, c := range []struct{ accel, in, want string }{
		{AccelGPU, "cuda", "cuda"}, {AccelGPU, "openvino", "openvino"}, {AccelGPU, "directml", ""}, {AccelNPU, "openvino", "openvino"},
		{AccelGPU, "tensorrt", ""}, {AccelGPU, "", ""}, {AccelCPU, "cuda", ""},
	} {
		if got := readyProvider(c.accel, c.in); got != c.want {
			t.Fatalf("readyProvider(%q, %q)=%q，要 %q", c.accel, c.in, got, c.want)
		}
	}
}
