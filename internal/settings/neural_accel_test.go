package settings

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// 硬體加速：預設 cpu、缺欄位時也是；三個值都能存讀；不認得的值退回 cpu 但原文保留。
func TestNeuralAccelDefaultsSetAndRoundTrip(t *testing.T) {
	if Default().NeuralAccel != AccelCPU {
		t.Fatalf("預設要是 cpu：%q", Default().NeuralAccel)
	}
	path := filepath.Join(t.TempDir(), FileName)
	os.WriteFile(path, []byte(`{"selection":"neural"}`), 0o600)
	s, err := Load(path)
	if err != nil || s.NeuralAccel != AccelCPU {
		t.Fatalf("缺欄位要是 cpu：%+v %v", s, err)
	}
	for _, v := range []string{AccelGPU, AccelNPU, AccelCPU} {
		if err := s.Set("neural-accel", v); err != nil {
			t.Fatalf("set %s：%v", v, err)
		}
		if err := Save(path, s); err != nil {
			t.Fatal(err)
		}
		back, err := Load(path)
		if err != nil || back.NeuralAccel != v {
			t.Fatalf("%s 來回不一致：%q %v", v, back.NeuralAccel, err)
		}
		if b, _ := os.ReadFile(path); !strings.Contains(strings.ReplaceAll(string(b), " ", ""), `"neural_accel":"`+v+`"`) {
			t.Fatalf("檔案要寫 neural_accel：%s", b)
		}
	}
	if err := s.Set("neural-accel", "vulkan"); err == nil {
		t.Fatal("不認得的值要拒絕")
	}
}

func TestNeuralAccelUnknownValueKeptVerbatim(t *testing.T) {
	path := filepath.Join(t.TempDir(), FileName)
	os.WriteFile(path, []byte(`{"neural_accel":"tpu","punctuation":true}`), 0o600)
	r, err := Read(path)
	if err != nil || r.Settings.NeuralAccel != AccelCPU || !r.Settings.Punctuation {
		t.Fatalf("未知值退回 cpu，其餘照常讀取：%+v %v", r.Settings, err)
	}
	if len(r.Notes) != 1 || !strings.Contains(r.Notes[0], "neural-accel") {
		t.Fatalf("應說明哪個欄位退回預設：%q", r.Notes)
	}
	if err := Save(path, r.Settings); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(path); !strings.Contains(strings.ReplaceAll(string(b), " ", ""), `"neural_accel":"tpu"`) {
		t.Fatalf("新版的值存檔時要原樣保留：%s", b)
	}
}

// 只用統計引擎時不載入模型，硬體加速停用，並指向「選字方式」。
func TestNeuralAccelDisabledWithStatisticalSelection(t *testing.T) {
	s := Default()
	s.Selection = Statistical
	if reason := s.disabledReasonOn("linux", "neural-accel"); reason != accelStatisticalReason || disabledBy(reason) != "selection" {
		t.Fatalf("統計引擎時要停用並指向 selection：%q", reason)
	}
	if s.Selection = Neural; s.disabledReasonOn("linux", "neural-accel") != "" {
		t.Fatal("神經選字時不停用")
	}
}

// macOS 沒有這一項，設定檔裡的值也不作用；Windows、Linux 照設定。
func TestNeuralAccelPlatforms(t *testing.T) {
	if NeuralAccelSetting("darwin") || !NeuralAccelSetting("linux") || !NeuralAccelSetting("windows") {
		t.Fatal("只有 Windows、Linux 有硬體加速")
	}
	s := Default()
	s.NeuralAccel = AccelGPU
	want := AccelGPU
	if !NeuralAccelSetting(runtime.GOOS) {
		want = AccelCPU
	}
	if got := s.EffectiveNeuralAccel(); got != want {
		t.Fatalf("EffectiveNeuralAccel=%q，要 %q", got, want)
	}
	listed := false
	for _, it := range s.Items() {
		listed = listed || it.Key == "neural-accel"
	}
	if listed != NeuralAccelSetting(runtime.GOOS) {
		t.Fatalf("Items 列出 neural-accel=%v，平台 %s", listed, runtime.GOOS)
	}
}
