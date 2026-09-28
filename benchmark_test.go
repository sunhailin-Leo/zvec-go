//go:build integration

package zvec

import (
	"fmt"
	"testing"
)

// Helper function to create a benchmark schema
func benchmarkCreateSchema(dimension uint32) *CollectionSchema {
	schema := NewCollectionSchema("bench_collection")

	invertParams := NewInvertIndexParams(true, false)
	hnswParams := NewHNSWIndexParams(MetricTypeCosine, 16, 200)

	idField := NewFieldSchema("id", DataTypeString, false, 0)
	_ = idField.SetIndexParams(invertParams)
	_ = schema.AddField(idField)

	embField := NewFieldSchema("embedding", DataTypeVectorFP32, false, dimension)
	_ = embField.SetIndexParams(hnswParams)
	_ = schema.AddField(embField)

	return schema
}

// Helper function to generate a random vector
func generateRandomVector(dimension int) []float32 {
	vec := make([]float32, dimension)
	for i := range vec {
		vec[i] = float32(i) * 0.01
	}
	return vec
}

// Schema-related benchmarks

func BenchmarkNewCollectionSchema(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		schema := NewCollectionSchema("bench_collection")
		schema.Destroy()
	}
}

func BenchmarkNewFieldSchema(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		field := NewFieldSchema("test_field", DataTypeString, false, 0)
		field.Destroy()
	}
}

func BenchmarkNewHNSWIndexParams(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		params := NewHNSWIndexParams(MetricTypeCosine, 16, 200)
		params.Destroy()
	}
}

// Doc-related benchmarks

func BenchmarkDocCreateDestroy(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		doc := NewDoc()
		doc.Destroy()
	}
}

// benchmarkFieldNames returns a fixed pool of field names so benchmarks can
// index into them (i % len) inside the timed loop without paying fmt.Sprintf
// allocations that would drown out the FFI cost under measurement.
func benchmarkFieldNames() []string {
	names := make([]string, 10)
	for i := range names {
		names[i] = fmt.Sprintf("field_%d", i)
	}
	return names
}

func BenchmarkDocSetPK(b *testing.B) {
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	pks := make([]string, 1024)
	for i := range pks {
		pks[i] = fmt.Sprintf("doc_%d", i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doc.SetPK(pks[i%len(pks)])
	}
}

func BenchmarkDocAddStringField(b *testing.B) {
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	fieldNames := benchmarkFieldNames()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = doc.AddStringField(fieldNames[i%len(fieldNames)], "test value")
	}
}

func benchmarkDocAddVectorFP32Field(b *testing.B, dimension int) {
	b.Helper()
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	fieldNames := benchmarkFieldNames()
	vector := generateRandomVector(dimension)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = doc.AddVectorFP32Field(fieldNames[i%len(fieldNames)], vector)
	}
}

func BenchmarkDocAddVectorFP32Field_4D(b *testing.B) {
	benchmarkDocAddVectorFP32Field(b, 4)
}

func BenchmarkDocAddVectorFP32Field_128D(b *testing.B) {
	benchmarkDocAddVectorFP32Field(b, 128)
}

func BenchmarkDocAddVectorFP32Field_768D(b *testing.B) {
	benchmarkDocAddVectorFP32Field(b, 768)
}

func BenchmarkDocGetStringField(b *testing.B) {
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	fieldNames := benchmarkFieldNames()

	// Pre-populate with fields
	for i := 0; i < 100; i++ {
		_ = doc.AddStringField(fieldNames[i%len(fieldNames)], "test value")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = doc.GetStringField(fieldNames[i%len(fieldNames)])
	}
}

func BenchmarkDocGetVectorFP32Field_128D(b *testing.B) {
	b.ReportAllocs()
	doc := NewDoc()
	defer doc.Destroy()

	// Pre-populate with vector field
	vector := generateRandomVector(128)
	_ = doc.AddVectorFP32Field("embedding", vector)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = doc.GetVectorFP32Field("embedding")
	}
}

// Query-related benchmarks

func BenchmarkNewVectorQuery(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		query := NewVectorQuery()
		query.Destroy()
	}
}

func BenchmarkVectorQuerySetup(b *testing.B) {
	b.ReportAllocs()
	query := NewVectorQuery()
	defer query.Destroy()

	queryVector := generateRandomVector(128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = query.SetFieldName("embedding")
		_ = query.SetTopK(10)
		_ = query.SetQueryVector(queryVector)
		_ = query.SetFilter("id > 0")
	}
}

// Collection-related benchmarks

// benchmarkBuildInsertDoc builds a doc with the canonical insert shape
// (PK + string id field + 128-dim embedding vector). It allocates and pays
// several cgo transitions, so call sites exclude it from the measured region
// with StopTimer/StartTimer.
func benchmarkBuildInsertDoc(pk string) *Doc {
	doc := NewDoc()
	doc.SetPK(pk)
	_ = doc.AddStringField("id", pk)
	_ = doc.AddVectorFP32Field("embedding", generateRandomVector(128))
	return doc
}

// The Insert benchmarks below insert unique-PK documents every iteration, so
// the HNSW index keeps growing for the whole run. The final collection size
// therefore depends on b.N and thus on the machine; it is reported via the
// docs-inserted metric so numbers from different runs can be compared at a
// known corpus size. Doc construction is excluded from the timed region so
// ns/op isolates the collection.Insert FFI call and native index work.

func BenchmarkCollectionInsert(b *testing.B) {
	b.ReportAllocs()

	tmpDir := b.TempDir()
	path := tmpDir + "/col"
	schema := benchmarkCreateSchema(128)
	defer schema.Destroy()

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}
	defer func() { _ = collection.Close() }()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		doc := benchmarkBuildInsertDoc(fmt.Sprintf("doc_%d", i))
		b.StartTimer()

		_, err := collection.Insert([]*Doc{doc})
		if err != nil {
			b.Fatalf("Failed to insert document: %v", err)
		}

		b.StopTimer()
		doc.Destroy()
		b.StartTimer()
	}
	b.ReportMetric(float64(b.N), "docs-inserted")
}

func benchmarkCollectionInsertBatch(b *testing.B, batchSize int) {
	b.Helper()
	b.ReportAllocs()

	tmpDir := b.TempDir()
	path := tmpDir + "/col"
	schema := benchmarkCreateSchema(128)
	defer schema.Destroy()

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}
	defer func() { _ = collection.Close() }()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		docs := make([]*Doc, batchSize)
		for j := 0; j < batchSize; j++ {
			docs[j] = benchmarkBuildInsertDoc(fmt.Sprintf("doc_%d_%d", i, j))
		}
		b.StartTimer()

		_, err := collection.Insert(docs)
		if err != nil {
			b.Fatalf("Failed to insert documents: %v", err)
		}

		b.StopTimer()
		FreeDocs(docs)
		b.StartTimer()
	}
	b.ReportMetric(float64(b.N)*float64(batchSize), "docs-inserted")
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(batchSize), "ns/doc")
}

func BenchmarkCollectionInsertBatch10(b *testing.B) {
	benchmarkCollectionInsertBatch(b, 10)
}

func BenchmarkCollectionInsertBatch100(b *testing.B) {
	benchmarkCollectionInsertBatch(b, 100)
}

func BenchmarkCollectionQuery(b *testing.B) {
	b.ReportAllocs()

	tmpDir := b.TempDir()
	path := tmpDir + "/col"
	schema := benchmarkCreateSchema(128)
	defer schema.Destroy()

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}
	defer func() { _ = collection.Close() }()

	// Pre-insert 1000 documents
	b.StopTimer()
	for i := 0; i < 1000; i++ {
		doc := NewDoc()
		pk := fmt.Sprintf("doc_%d", i)
		doc.SetPK(pk)
		_ = doc.AddStringField("id", pk)
		vector := generateRandomVector(128)
		_ = doc.AddVectorFP32Field("embedding", vector)

		_, err := collection.Insert([]*Doc{doc})
		if err != nil {
			b.Fatalf("Failed to insert document: %v", err)
		}
		doc.Destroy()
	}

	if err := collection.Flush(); err != nil {
		b.Fatalf("Failed to flush collection: %v", err)
	}

	// Benchmark query
	query := NewVectorQuery()
	defer query.Destroy()
	if err := query.SetFieldName("embedding"); err != nil {
		b.Fatalf("Failed to set query field name: %v", err)
	}
	if err := query.SetTopK(10); err != nil {
		b.Fatalf("Failed to set query top K: %v", err)
	}
	queryVector := generateRandomVector(128)
	if err := query.SetQueryVector(queryVector); err != nil {
		b.Fatalf("Failed to set query vector: %v", err)
	}

	b.ResetTimer()
	b.StartTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		docs, err := collection.Query(query)
		if err != nil {
			b.Fatalf("Failed to query collection: %v", err)
		}
		FreeDocs(docs)
	}
	b.StopTimer()
}

func BenchmarkCollectionFetch(b *testing.B) {
	b.ReportAllocs()

	tmpDir := b.TempDir()
	path := tmpDir + "/col"
	schema := benchmarkCreateSchema(128)
	defer schema.Destroy()

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}
	defer func() { _ = collection.Close() }()

	// Pre-insert 100 documents
	b.StopTimer()
	for i := 0; i < 100; i++ {
		doc := NewDoc()
		pk := fmt.Sprintf("doc_%d", i)
		doc.SetPK(pk)
		_ = doc.AddStringField("id", pk)
		vector := generateRandomVector(128)
		_ = doc.AddVectorFP32Field("embedding", vector)

		_, err := collection.Insert([]*Doc{doc})
		if err != nil {
			b.Fatalf("Failed to insert document: %v", err)
		}
		doc.Destroy()
	}

	if err := collection.Flush(); err != nil {
		b.Fatalf("Failed to flush collection: %v", err)
	}

	// Benchmark fetch
	primaryKeys := make([][]string, 100)
	for index := range primaryKeys {
		primaryKeys[index] = []string{fmt.Sprintf("doc_%d", index)}
	}
	b.ResetTimer()
	b.StartTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		docs, err := collection.Fetch(primaryKeys[iteration%len(primaryKeys)])
		if err != nil {
			b.Fatalf("Failed to fetch document: %v", err)
		}
		FreeDocs(docs)
	}
	b.StopTimer()
}
