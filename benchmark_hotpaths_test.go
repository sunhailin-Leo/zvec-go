//go:build integration

package zvec

import (
	"fmt"
	"testing"
)

func benchmarkInsertDocs(benchmark *testing.B, collection *Collection, dimension, count int) []string {
	benchmark.Helper()
	primaryKeys := make([]string, count)
	docs := make([]*Doc, count)
	for index := range docs {
		primaryKeys[index] = fmt.Sprintf("doc_%d", index)
		doc := NewDoc()
		if doc == nil {
			FreeDocs(docs)
			benchmark.Fatal("NewDoc() returned nil")
		}
		docs[index] = doc
		doc.SetPK(primaryKeys[index])
		if err := doc.AddStringField("id", primaryKeys[index]); err != nil {
			FreeDocs(docs)
			benchmark.Fatalf("AddStringField() failed: %v", err)
		}
		vector := generateRandomVector(dimension)
		vector[0] = float32(index%16) * 0.01
		if err := doc.AddVectorFP32Field("embedding", vector); err != nil {
			FreeDocs(docs)
			benchmark.Fatalf("AddVectorFP32Field() failed: %v", err)
		}
	}
	result, err := collection.Insert(docs)
	FreeDocs(docs)
	if err != nil {
		benchmark.Fatalf("Insert() failed: %v", err)
	}
	if result.ErrorCount != 0 {
		benchmark.Fatalf("Insert() reported %d document errors", result.ErrorCount)
	}
	if err := collection.Flush(); err != nil {
		benchmark.Fatalf("Flush() failed: %v", err)
	}
	return primaryKeys
}

func BenchmarkCollectionQueryMatrix(benchmark *testing.B) {
	for _, dimension := range []int{128, 768} {
		benchmark.Run(fmt.Sprintf("Dim%d", dimension), func(dimensionBenchmark *testing.B) {
			schema := benchmarkCreateSchema(uint32(dimension))
			defer schema.Destroy()
			collection, err := CreateAndOpen(dimensionBenchmark.TempDir()+"/col", schema, nil)
			if err != nil {
				dimensionBenchmark.Fatalf("CreateAndOpen() failed: %v", err)
			}
			defer func() { _ = collection.Close() }()
			benchmarkInsertDocs(dimensionBenchmark, collection, dimension, 256)

			for _, topK := range []int{1, 10, 100} {
				dimensionBenchmark.Run(fmt.Sprintf("TopK%d", topK), func(queryBenchmark *testing.B) {
					query := NewVectorQuery()
					defer query.Destroy()
					if err := query.SetFieldName("embedding"); err != nil {
						queryBenchmark.Fatalf("SetFieldName() failed: %v", err)
					}
					if err := query.SetTopK(topK); err != nil {
						queryBenchmark.Fatalf("SetTopK() failed: %v", err)
					}
					if err := query.SetQueryVector(generateRandomVector(dimension)); err != nil {
						queryBenchmark.Fatalf("SetQueryVector() failed: %v", err)
					}
					for warmup := 0; warmup < 32; warmup++ {
						docs, err := collection.Query(query)
						if err != nil {
							queryBenchmark.Fatalf("Query() warmup failed: %v", err)
						}
						FreeDocs(docs)
					}
					queryBenchmark.ReportAllocs()
					queryBenchmark.ResetTimer()
					for iteration := 0; iteration < queryBenchmark.N; iteration++ {
						docs, err := collection.Query(query)
						if err != nil {
							queryBenchmark.Fatalf("Query() failed: %v", err)
						}
						FreeDocs(docs)
					}
				})
			}
		})
	}
}

func BenchmarkCollectionReopenAndFirstQuery(benchmark *testing.B) {
	schema := benchmarkCreateSchema(128)
	defer schema.Destroy()
	path := benchmark.TempDir() + "/col"
	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		benchmark.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer func() {
		if collection != nil {
			_ = collection.Close()
		}
	}()
	benchmarkInsertDocs(benchmark, collection, 128, 256)
	if err := collection.Close(); err != nil {
		benchmark.Fatalf("Close() failed: %v", err)
	}

	query := NewVectorQuery()
	defer query.Destroy()
	if err := query.SetFieldName("embedding"); err != nil {
		benchmark.Fatalf("SetFieldName() failed: %v", err)
	}
	if err := query.SetTopK(10); err != nil {
		benchmark.Fatalf("SetTopK() failed: %v", err)
	}
	if err := query.SetQueryVector(generateRandomVector(128)); err != nil {
		benchmark.Fatalf("SetQueryVector() failed: %v", err)
	}

	benchmark.ReportAllocs()
	benchmark.ResetTimer()
	for iteration := 0; iteration < benchmark.N; iteration++ {
		collection, err = Open(path, nil)
		if err != nil {
			benchmark.Fatalf("Open() failed: %v", err)
		}
		docs, err := collection.Query(query)
		if err != nil {
			benchmark.Fatalf("Query() failed: %v", err)
		}
		if len(docs) != 10 {
			FreeDocs(docs)
			benchmark.Fatalf("Query() returned %d documents, want 10", len(docs))
		}
		FreeDocs(docs)
		if err := collection.Close(); err != nil {
			benchmark.Fatalf("Close() failed: %v", err)
		}
	}
}

func BenchmarkCollectionFetchBatch(benchmark *testing.B) {
	schema := benchmarkCreateSchema(128)
	defer schema.Destroy()
	collection, err := CreateAndOpen(benchmark.TempDir()+"/col", schema, nil)
	if err != nil {
		benchmark.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer func() { _ = collection.Close() }()
	primaryKeys := benchmarkInsertDocs(benchmark, collection, 128, 1000)
	for _, count := range []int{1, 10, 100, 1000} {
		benchmark.Run(fmt.Sprintf("Keys%d", count), func(fetchBenchmark *testing.B) {
			keys := primaryKeys[:count]
			for warmup := 0; warmup < 32; warmup++ {
				docs, err := collection.Fetch(keys)
				if err != nil {
					fetchBenchmark.Fatalf("Fetch() warmup failed: %v", err)
				}
				FreeDocs(docs)
			}
			fetchBenchmark.ReportAllocs()
			fetchBenchmark.ResetTimer()
			for iteration := 0; iteration < fetchBenchmark.N; iteration++ {
				docs, err := collection.Fetch(keys)
				if err != nil {
					fetchBenchmark.Fatalf("Fetch() failed: %v", err)
				}
				FreeDocs(docs)
			}
		})
	}
}

func BenchmarkDocGetVectorFP32Field_768D(benchmark *testing.B) {
	doc := NewDoc()
	defer doc.Destroy()
	if err := doc.AddVectorFP32Field("embedding", generateRandomVector(768)); err != nil {
		benchmark.Fatalf("AddVectorFP32Field() failed: %v", err)
	}
	benchmark.ReportAllocs()
	benchmark.ResetTimer()
	for iteration := 0; iteration < benchmark.N; iteration++ {
		vector, err := doc.GetVectorFP32Field("embedding")
		if err != nil {
			benchmark.Fatalf("GetVectorFP32Field() failed: %v", err)
		}
		if len(vector) != 768 {
			benchmark.Fatalf("GetVectorFP32Field() returned %d dimensions, want 768", len(vector))
		}
	}
}

func benchmarkDocGetVectorFP32FieldInto(benchmark *testing.B, dimension int) {
	benchmark.Helper()
	doc := NewDoc()
	defer doc.Destroy()
	if err := doc.AddVectorFP32Field("embedding", generateRandomVector(dimension)); err != nil {
		benchmark.Fatalf("AddVectorFP32Field() failed: %v", err)
	}
	buffer := make([]float32, dimension)
	benchmark.ReportAllocs()
	benchmark.ResetTimer()
	for iteration := 0; iteration < benchmark.N; iteration++ {
		vector, err := doc.GetVectorFP32FieldInto("embedding", buffer)
		if err != nil {
			benchmark.Fatalf("GetVectorFP32FieldInto() failed: %v", err)
		}
		if len(vector) != dimension {
			benchmark.Fatalf("GetVectorFP32FieldInto() returned %d dimensions, want %d", len(vector), dimension)
		}
	}
}

func BenchmarkDocGetVectorFP32FieldInto_128D(benchmark *testing.B) {
	benchmarkDocGetVectorFP32FieldInto(benchmark, 128)
}

func BenchmarkDocGetVectorFP32FieldInto_768D(benchmark *testing.B) {
	benchmarkDocGetVectorFP32FieldInto(benchmark, 768)
}
