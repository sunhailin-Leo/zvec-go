//go:build integration

package zvec

import (
	"fmt"
	"testing"
)

func BenchmarkDocGetVectorFP32FieldComparison(benchmark *testing.B) {
	for _, dimension := range []int{128, 768} {
		benchmark.Run(fmt.Sprintf("Dim%d", dimension), func(dimensionBenchmark *testing.B) {
			document := NewDoc()
			if document == nil {
				dimensionBenchmark.Fatal("NewDoc() returned nil")
			}
			defer document.Destroy()
			if err := document.AddVectorFP32Field("embedding", generateRandomVector(dimension)); err != nil {
				dimensionBenchmark.Fatalf("AddVectorFP32Field() failed: %v", err)
			}
			dimensionBenchmark.Run("Allocating", func(getBenchmark *testing.B) {
				getBenchmark.ReportAllocs()
				getBenchmark.ResetTimer()
				for iteration := 0; iteration < getBenchmark.N; iteration++ {
					vector, err := document.GetVectorFP32Field("embedding")
					if err != nil || len(vector) != dimension {
						getBenchmark.Fatalf("GetVectorFP32Field() returned %d dimensions, err=%v", len(vector), err)
					}
				}
			})
			dimensionBenchmark.Run("Into", func(getBenchmark *testing.B) {
				buffer := make([]float32, dimension)
				getBenchmark.ReportAllocs()
				getBenchmark.ResetTimer()
				for iteration := 0; iteration < getBenchmark.N; iteration++ {
					vector, err := document.GetVectorFP32FieldInto("embedding", buffer)
					if err != nil || len(vector) != dimension {
						getBenchmark.Fatalf("GetVectorFP32FieldInto() returned %d dimensions, err=%v", len(vector), err)
					}
				}
			})
		})
	}
}
