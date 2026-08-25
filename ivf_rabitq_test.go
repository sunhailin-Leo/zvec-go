//go:build integration

package zvec

import (
	"testing"
)

// =============================================================================
// IVF RaBitQ IndexParams Tests
// =============================================================================

func TestNewIVFRaBitQIndexParams(t *testing.T) {
	params, err := NewIVFRaBitQIndexParams(MetricTypeL2, 16, 256, 0)
	if err != nil {
		t.Fatalf("NewIVFRaBitQIndexParams() failed: %v", err)
	}
	defer params.Destroy()

	if got := params.GetType(); got != IndexTypeIVFRaBitQ {
		t.Errorf("GetType() = %v, want %v", got, IndexTypeIVFRaBitQ)
	}

	if got := params.GetMetricType(); got != MetricTypeL2 {
		t.Errorf("GetMetricType() = %v, want %v", got, MetricTypeL2)
	}

	nlist, totalBits, sampleCount, err := params.GetIVFRaBitQParams()
	if err != nil {
		t.Fatalf("GetIVFRaBitQParams() failed: %v", err)
	}
	if nlist != 16 || totalBits != 256 || sampleCount != 0 {
		t.Errorf("GetIVFRaBitQParams() = (%d, %d, %d), want (16, 256, 0)",
			nlist, totalBits, sampleCount)
	}
}

func TestIVFRaBitQIndexParamsSetters(t *testing.T) {
	params, err := NewIVFRaBitQIndexParams(MetricTypeCosine, 8, 128, 0)
	if err != nil {
		t.Fatalf("NewIVFRaBitQIndexParams() failed: %v", err)
	}
	defer params.Destroy()

	if err := params.SetIVFRaBitQParams(32, 512, 1000); err != nil {
		t.Fatalf("SetIVFRaBitQParams() failed: %v", err)
	}
	nlist, totalBits, sampleCount, err := params.GetIVFRaBitQParams()
	if err != nil {
		t.Fatalf("GetIVFRaBitQParams() failed: %v", err)
	}
	if nlist != 32 || totalBits != 512 || sampleCount != 1000 {
		t.Errorf("GetIVFRaBitQParams() = (%d, %d, %d), want (32, 512, 1000)",
			nlist, totalBits, sampleCount)
	}
}

func TestIVFRaBitQIndexParamsDestroy(t *testing.T) {
	params, err := NewIVFRaBitQIndexParams(MetricTypeL2, 16, 256, 0)
	if err != nil {
		t.Fatalf("NewIVFRaBitQIndexParams() failed: %v", err)
	}

	// First Destroy should not panic
	params.Destroy()

	// Second Destroy should also not panic
	params.Destroy()
}

func TestIndexTypeIVFRaBitQString(t *testing.T) {
	if got := IndexTypeIVFRaBitQ.String(); got != "IVF_RABITQ" {
		t.Errorf("IndexTypeIVFRaBitQ.String() = %q, want %q", got, "IVF_RABITQ")
	}
}

func TestQuantizeTypeRABITQ(t *testing.T) {
	if QuantizeTypeRABITQ != 4 {
		t.Errorf("QuantizeTypeRABITQ = %d, want 4", QuantizeTypeRABITQ)
	}
	if got := QuantizeTypeRABITQ.String(); got != "RABITQ" {
		t.Errorf("QuantizeTypeRABITQ.String() = %q, want %q", got, "RABITQ")
	}
}

// =============================================================================
// Vamana two-pass build Tests
// =============================================================================

func TestVamanaTwoPassBuild(t *testing.T) {
	params := NewIndexParams(IndexTypeVamana)
	if params == nil {
		t.Fatal("NewIndexParams(IndexTypeVamana) returned nil")
	}
	defer params.Destroy()

	if got := params.GetType(); got != IndexTypeVamana {
		t.Errorf("GetType() = %v, want %v", got, IndexTypeVamana)
	}

	if err := params.SetVamanaTwoPassBuild(true); err != nil {
		t.Fatalf("SetVamanaTwoPassBuild(true) failed: %v", err)
	}
	if got := params.GetVamanaTwoPassBuild(); got != true {
		t.Errorf("GetVamanaTwoPassBuild() = %v, want true", got)
	}

	if err := params.SetVamanaTwoPassBuild(false); err != nil {
		t.Fatalf("SetVamanaTwoPassBuild(false) failed: %v", err)
	}
	if got := params.GetVamanaTwoPassBuild(); got != false {
		t.Errorf("GetVamanaTwoPassBuild() = %v, want false", got)
	}
}

// =============================================================================
// IVFRaBitQQueryParams Tests
// =============================================================================

func TestNewIVFRaBitQQueryParams(t *testing.T) {
	params := NewIVFRaBitQQueryParams(20, 0.5, false, true)
	if params == nil {
		t.Fatal("NewIVFRaBitQQueryParams returned nil")
	}
	defer params.Destroy()

	if got := params.GetNprobe(); got != 20 {
		t.Errorf("GetNprobe() = %d, want %d", got, 20)
	}
	if got := params.GetRadius(); got != 0.5 {
		t.Errorf("GetRadius() = %f, want %f", got, 0.5)
	}
	if got := params.GetIsLinear(); got != false {
		t.Errorf("GetIsLinear() = %v, want false", got)
	}
	if got := params.GetIsUsingRefiner(); got != true {
		t.Errorf("GetIsUsingRefiner() = %v, want true", got)
	}
}

func TestIVFRaBitQQueryParamsSetters(t *testing.T) {
	params := NewIVFRaBitQQueryParams(10, 0.0, false, false)
	if params == nil {
		t.Fatal("NewIVFRaBitQQueryParams returned nil")
	}
	defer params.Destroy()

	if err := params.SetNprobe(64); err != nil {
		t.Errorf("SetNprobe failed: %v", err)
	}
	if got := params.GetNprobe(); got != 64 {
		t.Errorf("GetNprobe() = %d, want %d", got, 64)
	}

	if err := params.SetScaleFactor(2.5); err != nil {
		t.Errorf("SetScaleFactor failed: %v", err)
	}
	if got := params.GetScaleFactor(); got != 2.5 {
		t.Errorf("GetScaleFactor() = %f, want %f", got, 2.5)
	}

	if err := params.SetRadius(1.25); err != nil {
		t.Errorf("SetRadius failed: %v", err)
	}
	if got := params.GetRadius(); got != 1.25 {
		t.Errorf("GetRadius() = %f, want %f", got, 1.25)
	}

	if err := params.SetIsLinear(true); err != nil {
		t.Errorf("SetIsLinear failed: %v", err)
	}
	if got := params.GetIsLinear(); got != true {
		t.Errorf("GetIsLinear() = %v, want true", got)
	}

	if err := params.SetIsUsingRefiner(true); err != nil {
		t.Errorf("SetIsUsingRefiner failed: %v", err)
	}
	if got := params.GetIsUsingRefiner(); got != true {
		t.Errorf("GetIsUsingRefiner() = %v, want true", got)
	}
}

func TestIVFRaBitQQueryParamsDestroy(t *testing.T) {
	params := NewIVFRaBitQQueryParams(10, 0.0, false, false)
	if params == nil {
		t.Fatal("NewIVFRaBitQQueryParams returned nil")
	}

	// First Destroy should not panic
	params.Destroy()

	// Second Destroy should also not panic
	params.Destroy()
}

// =============================================================================
// SetIVFRaBitQParams ownership transfer Tests
// =============================================================================

func TestSearchQuerySetIVFRaBitQParams(t *testing.T) {
	query := NewSearchQuery()
	if query == nil {
		t.Fatal("NewSearchQuery returned nil")
	}
	defer query.Destroy()

	if err := query.SetFieldName("vector_field"); err != nil {
		t.Fatalf("SetFieldName failed: %v", err)
	}

	params := NewIVFRaBitQQueryParams(10, 0.0, false, false)
	if params == nil {
		t.Fatal("NewIVFRaBitQQueryParams returned nil")
	}
	if err := query.SetIVFRaBitQParams(params); err != nil {
		t.Fatalf("SetIVFRaBitQParams failed: %v", err)
	}
	if params.handle != nil {
		t.Fatal("SetIVFRaBitQParams did not transfer ownership")
	}

	// Destroy after ownership transfer should not panic
	params.Destroy()
}

func TestSearchQuerySetIVFRaBitQParamsNil(t *testing.T) {
	query := NewSearchQuery()
	if query == nil {
		t.Fatal("NewSearchQuery returned nil")
	}
	defer query.Destroy()

	if err := query.SetIVFRaBitQParams(nil); err == nil {
		t.Fatal("SetIVFRaBitQParams(nil) should fail")
	}
}

func TestGroupBySearchQuerySetIVFRaBitQParams(t *testing.T) {
	query := NewGroupBySearchQuery()
	if query == nil {
		t.Fatal("NewGroupBySearchQuery returned nil")
	}
	defer query.Destroy()

	if err := query.SetFieldName("vector_field"); err != nil {
		t.Fatalf("SetFieldName failed: %v", err)
	}

	params := NewIVFRaBitQQueryParams(10, 0.0, false, false)
	if params == nil {
		t.Fatal("NewIVFRaBitQQueryParams returned nil")
	}
	if err := query.SetIVFRaBitQParams(params); err != nil {
		t.Fatalf("SetIVFRaBitQParams failed: %v", err)
	}
	if params.handle != nil {
		t.Fatal("SetIVFRaBitQParams did not transfer ownership")
	}

	// Destroy after ownership transfer should not panic
	params.Destroy()
}

func TestSubQuerySetIVFRaBitQParams(t *testing.T) {
	subQuery := NewSubQuery()
	if subQuery == nil {
		t.Fatal("NewSubQuery returned nil")
	}
	defer subQuery.Destroy()

	if err := subQuery.SetFieldName("vector_field"); err != nil {
		t.Fatalf("SetFieldName failed: %v", err)
	}

	params := NewIVFRaBitQQueryParams(10, 0.0, false, false)
	if params == nil {
		t.Fatal("NewIVFRaBitQQueryParams returned nil")
	}
	if err := subQuery.SetIVFRaBitQParams(params); err != nil {
		t.Fatalf("SetIVFRaBitQParams failed: %v", err)
	}
	if params.handle != nil {
		t.Fatal("SetIVFRaBitQParams did not transfer ownership")
	}

	// Destroy after ownership transfer should not panic
	params.Destroy()
}
