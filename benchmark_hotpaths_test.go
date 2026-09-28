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

// BenchmarkCgoCallFloor_GetScore measures the irreducible cost of ONE cgo
// transition: GetScore is the cheapest native call in the SDK (single
// transition, no argument marshaling, no native allocation). Divide any
// other benchmark's transition count by this floor to see its unavoidable
// FFI share: ns/op ≈ transitions × floor + native work.
func BenchmarkCgoCallFloor_GetScore(benchmark *testing.B) {
	doc := NewDoc()
	defer doc.Destroy()
	doc.SetScore(1.5)

	benchmark.ReportAllocs()
	for warmup := 0; warmup < 32; warmup++ {
		_ = doc.GetScore()
	}
	benchmark.ResetTimer()
	for iteration := 0; iteration < benchmark.N; iteration++ {
		_ = doc.GetScore()
	}
}

// BenchmarkCgoCallFloor_IsInitialized is a handle-free variant of the cgo
// transition floor probe.
func BenchmarkCgoCallFloor_IsInitialized(benchmark *testing.B) {
	benchmark.ReportAllocs()
	for warmup := 0; warmup < 32; warmup++ {
		_ = IsInitialized()
	}
	benchmark.ResetTimer()
	for iteration := 0; iteration < benchmark.N; iteration++ {
		_ = IsInitialized()
	}
}

// benchmarkDecodeResultDoc reads every per-result value a real consumer of a
// search wants: the PK, the score, and the vector (into a shared buffer, so
// vector decoding itself is allocation-free).
func benchmarkDecodeResultDoc(benchmark *testing.B, doc *Doc, buffer []float32) {
	if doc.GetPK() == "" {
		benchmark.Fatal("GetPK() returned empty primary key")
	}
	_ = doc.GetScore()
	if _, err := doc.GetVectorFP32FieldInto("embedding", buffer); err != nil {
		benchmark.Fatalf("GetVectorFP32FieldInto() failed: %v", err)
	}
}

// BenchmarkCollectionQueryResultDecode measures the full user-visible search
// round trip: Query + per-doc decode (PK, score, vector into a shared
// buffer) + FreeDocs, all inside the timed region. The Query-only numbers
// from BenchmarkCollectionQueryMatrix deliberately skip decoding; comparing
// the two reveals how much of end-to-end search latency the decode path
// costs (GetPK is one Go allocation per doc and the next optimization
// target; vector decode through the Into API is zero-alloc).
func BenchmarkCollectionQueryResultDecode(benchmark *testing.B) {
	const dimension = 128

	schema := benchmarkCreateSchema(dimension)
	defer schema.Destroy()
	collection, err := CreateAndOpen(benchmark.TempDir()+"/col", schema, nil)
	if err != nil {
		benchmark.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer func() { _ = collection.Close() }()
	benchmarkInsertDocs(benchmark, collection, dimension, 256)

	decodeBuffer := make([]float32, dimension)
	for _, topK := range []int{10, 100} {
		benchmark.Run(fmt.Sprintf("TopK%d", topK), func(decodeBenchmark *testing.B) {
			query := NewVectorQuery()
			defer query.Destroy()
			if err := query.SetFieldName("embedding"); err != nil {
				decodeBenchmark.Fatalf("SetFieldName() failed: %v", err)
			}
			if err := query.SetTopK(topK); err != nil {
				decodeBenchmark.Fatalf("SetTopK() failed: %v", err)
			}
			if err := query.SetIncludeVector(true); err != nil {
				decodeBenchmark.Fatalf("SetIncludeVector() failed: %v", err)
			}
			if err := query.SetQueryVector(generateRandomVector(dimension)); err != nil {
				decodeBenchmark.Fatalf("SetQueryVector() failed: %v", err)
			}

			decodeDocs := func() {
				docs, err := collection.Query(query)
				if err != nil {
					decodeBenchmark.Fatalf("Query() failed: %v", err)
				}
				if len(docs) != topK {
					FreeDocs(docs)
					decodeBenchmark.Fatalf("Query() returned %d documents, want %d", len(docs), topK)
				}
				for _, doc := range docs {
					benchmarkDecodeResultDoc(decodeBenchmark, doc, decodeBuffer)
				}
				FreeDocs(docs)
			}
			for warmup := 0; warmup < 16; warmup++ {
				decodeDocs()
			}

			decodeBenchmark.ReportAllocs()
			decodeBenchmark.ResetTimer()
			for iteration := 0; iteration < decodeBenchmark.N; iteration++ {
				decodeDocs()
			}
			decodeBenchmark.ReportMetric(
				float64(decodeBenchmark.Elapsed().Nanoseconds())/float64(decodeBenchmark.N)/float64(topK),
				"ns/doc")
		})
	}
}

// benchmarkWriteCollection sets up a 256-doc collection (128-dim vectors)
// for write-path benchmarks. The returned schema must be kept alive (defer
// Destroy) for the duration of the benchmark, mirroring the other benches.
func benchmarkWriteCollection(benchmark *testing.B) (*CollectionSchema, *Collection, []string) {
	benchmark.Helper()

	schema := benchmarkCreateSchema(128)
	collection, err := CreateAndOpen(benchmark.TempDir()+"/col", schema, nil)
	if err != nil {
		benchmark.Fatalf("CreateAndOpen() failed: %v", err)
	}
	primaryKeys := benchmarkInsertDocs(benchmark, collection, 128, 256)
	return schema, collection, primaryKeys
}

// benchmarkBuildBatch rebuilds batchSize docs with the given primary keys so
// every iteration writes fresh document values over a stable set of keys.
// Callers exclude this (allocation + several cgo transitions) from the timed
// region with StopTimer/StartTimer.
func benchmarkBuildBatch(benchmark *testing.B, primaryKeys []string) []*Doc {
	benchmark.Helper()
	docs := make([]*Doc, len(primaryKeys))
	for index, primaryKey := range primaryKeys {
		docs[index] = benchmarkBuildInsertDoc(primaryKey)
		if docs[index] == nil {
			benchmark.Fatal("benchmarkBuildInsertDoc() returned nil")
		}
	}
	return docs
}

const benchmarkWriteBatchSize = 100

// BenchmarkCollectionUpdateBatch measures batched Update against a fixed
// corpus of 256 docs: documents for the same 100 keys are rebuilt outside
// the timed region, so ns/op isolates the Update FFI call + native write
// work (reported as ns/key).
func BenchmarkCollectionUpdateBatch(benchmark *testing.B) {
	schema, collection, primaryKeys := benchmarkWriteCollection(benchmark)
	defer schema.Destroy()
	defer func() { _ = collection.Close() }()
	keys := primaryKeys[:benchmarkWriteBatchSize]

	benchmark.ReportAllocs()
	benchmark.ResetTimer()
	for iteration := 0; iteration < benchmark.N; iteration++ {
		benchmark.StopTimer()
		docs := benchmarkBuildBatch(benchmark, keys)
		benchmark.StartTimer()

		if _, err := collection.Update(docs); err != nil {
			benchmark.Fatalf("Update() failed: %v", err)
		}

		benchmark.StopTimer()
		FreeDocs(docs)
		benchmark.StartTimer()
	}
	benchmark.ReportMetric(
		float64(benchmark.Elapsed().Nanoseconds())/float64(benchmark.N)/float64(benchmarkWriteBatchSize),
		"ns/key")
}

// BenchmarkCollectionUpsertBatch measures batched Upsert against a fixed
// corpus of 256 docs, same methodology as BenchmarkCollectionUpdateBatch.
func BenchmarkCollectionUpsertBatch(benchmark *testing.B) {
	schema, collection, primaryKeys := benchmarkWriteCollection(benchmark)
	defer schema.Destroy()
	defer func() { _ = collection.Close() }()
	keys := primaryKeys[:benchmarkWriteBatchSize]

	benchmark.ReportAllocs()
	benchmark.ResetTimer()
	for iteration := 0; iteration < benchmark.N; iteration++ {
		benchmark.StopTimer()
		docs := benchmarkBuildBatch(benchmark, keys)
		benchmark.StartTimer()

		if _, err := collection.Upsert(docs); err != nil {
			benchmark.Fatalf("Upsert() failed: %v", err)
		}

		benchmark.StopTimer()
		FreeDocs(docs)
		benchmark.StartTimer()
	}
	benchmark.ReportMetric(
		float64(benchmark.Elapsed().Nanoseconds())/float64(benchmark.N)/float64(benchmarkWriteBatchSize),
		"ns/key")
}

// BenchmarkCollectionDeleteBatch measures batched Delete of 100 live keys.
// Every iteration deletes the same key batch and restores the corpus with an
// Upsert outside the timed region, so each iteration measures the same
// fixed-size live-key delete path (not the missing-key path) at a constant
// collection size.
func BenchmarkCollectionDeleteBatch(benchmark *testing.B) {
	schema, collection, primaryKeys := benchmarkWriteCollection(benchmark)
	defer schema.Destroy()
	defer func() { _ = collection.Close() }()
	keys := primaryKeys[:benchmarkWriteBatchSize]

	restoreCorpus := func() {
		docs := benchmarkBuildBatch(benchmark, keys)
		if _, err := collection.Upsert(docs); err != nil {
			benchmark.Fatalf("Upsert() corpus restore failed: %v", err)
		}
		FreeDocs(docs)
	}

	benchmark.ReportAllocs()
	benchmark.ResetTimer()
	for iteration := 0; iteration < benchmark.N; iteration++ {
		if _, err := collection.Delete(keys); err != nil {
			benchmark.Fatalf("Delete() failed: %v", err)
		}

		benchmark.StopTimer()
		restoreCorpus()
		benchmark.StartTimer()
	}
	benchmark.ReportMetric(
		float64(benchmark.Elapsed().Nanoseconds())/float64(benchmark.N)/float64(benchmarkWriteBatchSize),
		"ns/key")
}

// BenchmarkCollectionQueryParallel exercises concurrent query throughput
// against one shared collection, the shape of an in-process vector service
// (each goroutine issues its own query object, per-request style). Nothing
// else in this package measures the SDK under goroutine concurrency, where
// cgo thread scheduling, native TLS error churn, and engine lock contention
// only appear. Note: native per-query thread counts (query thread count in
// the initialize config) are NOT varied here because reconfiguring the
// library's global init inside a benchmark process would disturb the other
// benchmarks; run separate bench invocations with GOMAXPROCS varied, or extend the
// Initialize config externally, to study that axis.
func BenchmarkCollectionQueryParallel(benchmark *testing.B) {
	const dimension = 128

	schema := benchmarkCreateSchema(dimension)
	defer schema.Destroy()
	collection, err := CreateAndOpen(benchmark.TempDir()+"/col", schema, nil)
	if err != nil {
		benchmark.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer func() { _ = collection.Close() }()
	benchmarkInsertDocs(benchmark, collection, dimension, 256)

	benchmark.ReportAllocs()
	benchmark.ResetTimer()
	benchmark.RunParallel(func(pb *testing.PB) {
		query := NewVectorQuery()
		defer query.Destroy()
		if err := query.SetFieldName("embedding"); err != nil {
			benchmark.Errorf("SetFieldName() failed: %v", err)
			return
		}
		if err := query.SetTopK(10); err != nil {
			benchmark.Errorf("SetTopK() failed: %v", err)
			return
		}
		if err := query.SetQueryVector(generateRandomVector(dimension)); err != nil {
			benchmark.Errorf("SetQueryVector() failed: %v", err)
			return
		}
		for pb.Next() {
			docs, err := collection.Query(query)
			if err != nil {
				benchmark.Errorf("Query() failed: %v", err)
				return
			}
			FreeDocs(docs)
		}
	})
}
