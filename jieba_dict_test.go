//go:build integration

package zvec

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFindJiebaDictDir(t *testing.T) {
	dir := FindJiebaDictDir()
	if dir == "" {
		t.Skip("no jieba dictionaries available in this environment")
	}
	for _, name := range []string{"jieba.dict.utf8", "hmm_model.utf8"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Errorf("FindJiebaDictDir() = %q, but %s is missing: %v", dir, name, err)
		}
	}
}

func TestAutoJiebaDictRegistration(t *testing.T) {
	wantDir := FindJiebaDictDir()
	if wantDir == "" {
		t.Skip("no jieba dictionaries available in this environment")
	}

	oldDefault := GetDefaultJiebaDictDir()
	t.Cleanup(func() { SetDefaultJiebaDictDir(oldDefault) })

	// An empty default is filled automatically.
	SetDefaultJiebaDictDir("")
	ensureDefaultJiebaDictDir()
	if got := GetDefaultJiebaDictDir(); got != wantDir {
		t.Errorf("ensureDefaultJiebaDictDir() set %q, want %q", got, wantDir)
	}

	// An explicitly configured default is preserved.
	explicitDir := t.TempDir()
	SetDefaultJiebaDictDir(explicitDir)
	ensureDefaultJiebaDictDir()
	if got := GetDefaultJiebaDictDir(); got != explicitDir {
		t.Errorf("ensureDefaultJiebaDictDir() overrode explicit dir: got %q, want %q", got, explicitDir)
	}

	// The opt-out env var disables auto-registration.
	SetDefaultJiebaDictDir("")
	t.Setenv(zvecDisableAutoJiebaDictEnv, "1")
	ensureDefaultJiebaDictDir()
	if got := GetDefaultJiebaDictDir(); got != "" {
		t.Errorf("ensureDefaultJiebaDictDir() with %s set: got %q, want empty", zvecDisableAutoJiebaDictEnv, got)
	}
}

// TestFTSJiebaTokenizerWithAutoDict exercises the `jieba` FTS tokenizer end
// to end without any manual dict configuration: Initialize (run by TestMain)
// must have auto-registered the bundled dictionaries.
func TestFTSJiebaTokenizerWithAutoDict(t *testing.T) {
	if FindJiebaDictDir() == "" {
		t.Skip("no jieba dictionaries available in this environment")
	}
	if got := GetDefaultJiebaDictDir(); got == "" {
		t.Fatal("GetDefaultJiebaDictDir() is empty after Initialize(); auto-registration did not run")
	}

	schema := NewCollectionSchema("jieba_auto_dict")
	if schema == nil {
		t.Fatal("schema stage: NewCollectionSchema() returned nil")
	}
	defer schema.Destroy()

	idField := NewFieldSchema("id", DataTypeString, false, 0)
	defer idField.Destroy()
	idIndex, err := NewInvertIndexParams(true, false)
	if err != nil {
		t.Fatalf("schema stage: NewInvertIndexParams(id) failed: %v", err)
	}
	defer idIndex.Destroy()
	if err := idField.SetIndexParams(idIndex); err != nil {
		t.Fatalf("schema stage: id.SetIndexParams() failed: %v", err)
	}
	if err := schema.AddField(idField); err != nil {
		t.Fatalf("schema stage: AddField(id) failed: %v", err)
	}

	contentField := NewFieldSchema("content", DataTypeString, false, 0)
	defer contentField.Destroy()
	ftsIndex, err := NewFTSIndexParams("jieba", nil, "")
	if err != nil {
		t.Fatalf("schema stage: NewFTSIndexParams(jieba) failed: %v", err)
	}
	defer ftsIndex.Destroy()
	if err := contentField.SetIndexParams(ftsIndex); err != nil {
		t.Fatalf("schema stage: content.SetIndexParams() failed: %v", err)
	}
	if err := schema.AddField(contentField); err != nil {
		t.Fatalf("schema stage: AddField(content) failed: %v", err)
	}

	vectorField := NewFieldSchema("embedding", DataTypeVectorFP32, false, 4)
	defer vectorField.Destroy()
	vectorIndex, err := NewFlatIndexParams(MetricTypeIP)
	if err != nil {
		t.Fatalf("schema stage: NewFlatIndexParams() failed: %v", err)
	}
	defer vectorIndex.Destroy()
	if err := vectorField.SetIndexParams(vectorIndex); err != nil {
		t.Fatalf("schema stage: embedding.SetIndexParams() failed: %v", err)
	}
	if err := schema.AddField(vectorField); err != nil {
		t.Fatalf("schema stage: AddField(embedding) failed: %v", err)
	}

	collection, err := CreateAndOpen(filepath.Join(t.TempDir(), "collection"), schema, nil)
	if err != nil {
		t.Fatalf("collection stage: CreateAndOpen() failed: %v", err)
	}
	defer func() {
		if err := collection.Close(); err != nil {
			t.Errorf("collection cleanup stage: Close() failed: %v", err)
		}
	}()

	samples := []struct {
		id      string
		content string
		vector  []float32
	}{
		{"doc1", "我爱自然语言处理技术", []float32{1.0, 0.0, 0.0, 0.0}},
		{"doc2", "向量数据库支持全文检索", []float32{0.0, 1.0, 0.0, 0.0}},
		{"doc3", "今天天气非常好", []float32{0.0, 0.0, 1.0, 0.0}},
	}
	docs := make([]*Doc, 0, len(samples))
	defer FreeDocs(docs)
	for _, sample := range samples {
		doc := NewDoc()
		if doc == nil {
			t.Fatalf("insert stage: NewDoc(%s) returned nil", sample.id)
		}
		docs = append(docs, doc)
		doc.SetPK(sample.id)
		if err := doc.AddStringField("id", sample.id); err != nil {
			t.Fatalf("insert stage: AddStringField(id) for %s failed: %v", sample.id, err)
		}
		if err := doc.AddStringField("content", sample.content); err != nil {
			t.Fatalf("insert stage: AddStringField(content) for %s failed: %v", sample.id, err)
		}
		if err := doc.AddVectorFP32Field("embedding", sample.vector); err != nil {
			t.Fatalf("insert stage: AddVectorFP32Field(embedding) for %s failed: %v", sample.id, err)
		}
	}

	insertResult, err := collection.Insert(docs)
	if err != nil {
		t.Fatalf("insert stage: Insert() failed: %v", err)
	}
	if insertResult.SuccessCount != uint64(len(samples)) || insertResult.ErrorCount != 0 {
		t.Fatalf(
			"insert stage: Insert() success=%d error=%d, want success=%d error=0",
			insertResult.SuccessCount, insertResult.ErrorCount, len(samples),
		)
	}
	if err := collection.Flush(); err != nil {
		t.Fatalf("flush stage: Flush() failed: %v", err)
	}

	query := NewSearchQuery()
	if query == nil {
		t.Fatal("query stage: NewSearchQuery() returned nil")
	}
	defer query.Destroy()
	if err := query.SetFieldName("content"); err != nil {
		t.Fatalf("query stage: SetFieldName() failed: %v", err)
	}
	if err := query.SetTopK(10); err != nil {
		t.Fatalf("query stage: SetTopK() failed: %v", err)
	}
	if err := query.SetOutputFields([]string{"id", "content"}); err != nil {
		t.Fatalf("query stage: SetOutputFields() failed: %v", err)
	}
	ftsPayload := NewFTS()
	if ftsPayload == nil {
		t.Fatal("query stage: NewFTS() returned nil")
	}
	defer ftsPayload.Destroy()
	if err := ftsPayload.SetMatchString("自然语言"); err != nil {
		t.Fatalf("query stage: SetMatchString() failed: %v", err)
	}
	if err := query.SetFTS(ftsPayload); err != nil {
		t.Fatalf("query stage: SetFTS() failed: %v", err)
	}

	results, err := collection.Query(query)
	if err != nil {
		t.Fatalf("query stage: Query() failed: %v", err)
	}
	defer FreeDocs(results)
	if len(results) != 1 {
		t.Fatalf("query stage: got %d results, want 1", len(results))
	}
	if pk := results[0].GetPK(); pk != "doc1" {
		t.Fatalf("query stage: got pk %q, want %q", pk, "doc1")
	}
}
