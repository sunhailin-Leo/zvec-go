//go:build integration

package zvec

import (
	"io"
	"path/filepath"
	"sort"
	"testing"
)

// newIteratorTestCollection creates a collection with docCount documents
// ("doc0".."docN-1") and flushes them to disk.
func newIteratorTestCollection(t *testing.T, docCount int) *Collection {
	t.Helper()

	schema := createTestSchema()
	defer schema.Destroy()

	path := filepath.Join(testTempDir(t), "iterator_collection")
	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	t.Cleanup(func() { _ = collection.Close() })

	docs := make([]*Doc, 0, docCount)
	for i := 0; i < docCount; i++ {
		pk := "doc" + string(rune('0'+i))
		docs = append(docs, createTestDoc(pk, "text "+pk, []float32{
			0.1 * float32(i), 0.2, 0.3, 0.4,
		}))
	}
	defer FreeDocs(docs)

	if _, err := collection.Insert(docs); err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}
	if err := collection.Flush(); err != nil {
		t.Fatalf("Flush() failed: %v", err)
	}
	return collection
}

func iteratorAllDocs(t *testing.T, iter *DocIterator) []*Doc {
	t.Helper()
	var docs []*Doc
	for {
		doc, err := iter.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			FreeDocs(docs)
			t.Fatalf("Next() failed: %v", err)
		}
		docs = append(docs, doc)
	}
	return docs
}

func TestCollectionCreateIteratorDefaults(t *testing.T) {
	const docCount = 5
	collection := newIteratorTestCollection(t, docCount)

	iter, err := collection.CreateIterator(nil)
	if err != nil {
		t.Fatalf("CreateIterator() failed: %v", err)
	}

	docs := iteratorAllDocs(t, iter)
	iter.Close()
	defer FreeDocs(docs)

	if len(docs) != docCount {
		t.Fatalf("iterator returned %d docs, want %d", len(docs), docCount)
	}

	seen := make(map[string]bool, docCount)
	for _, doc := range docs {
		seen[doc.GetPK()] = true
		if _, err := doc.GetStringField("id"); err != nil {
			t.Errorf("GetStringField(id) failed: %v", err)
		}
		// Default options include vectors.
		if !doc.HasField("embedding") {
			t.Errorf("doc %s missing vector field with default options", doc.GetPK())
		}
	}
	if len(seen) != docCount {
		t.Errorf("iterator returned %d distinct PKs, want %d", len(seen), docCount)
	}
}

func TestCollectionCreateIteratorWithOptions(t *testing.T) {
	const docCount = 3
	collection := newIteratorTestCollection(t, docCount)

	opts := NewIteratorOptions()
	if opts == nil {
		t.Fatal("NewIteratorOptions returned nil")
	}
	defer opts.Destroy()

	if err := opts.SetOutputFields([]string{"id"}); err != nil {
		t.Fatalf("SetOutputFields() failed: %v", err)
	}
	if err := opts.SetIncludeVector(false); err != nil {
		t.Fatalf("SetIncludeVector() failed: %v", err)
	}

	iter, err := collection.CreateIterator(opts)
	if err != nil {
		t.Fatalf("CreateIterator() failed: %v", err)
	}

	docs := iteratorAllDocs(t, iter)
	iter.Close()
	defer FreeDocs(docs)

	if len(docs) != docCount {
		t.Fatalf("iterator returned %d docs, want %d", len(docs), docCount)
	}
	for _, doc := range docs {
		if _, err := doc.GetStringField("id"); err != nil {
			t.Errorf("GetStringField(id) failed: %v", err)
		}
		// "text" was not requested.
		if doc.HasField("text") {
			t.Errorf("doc %s unexpectedly contains excluded field text", doc.GetPK())
		}
		if doc.HasField("embedding") {
			t.Errorf("doc %s unexpectedly contains vector with include_vector=false", doc.GetPK())
		}
	}
}

func TestCollectionCreateIteratorEmptyOutputFields(t *testing.T) {
	collection := newIteratorTestCollection(t, 2)

	opts := NewIteratorOptions()
	if opts == nil {
		t.Fatal("NewIteratorOptions returned nil")
	}
	defer opts.Destroy()

	// Non-nil empty slice: no scalar fields, only PK / system columns.
	if err := opts.SetOutputFields([]string{}); err != nil {
		t.Fatalf("SetOutputFields() failed: %v", err)
	}
	if err := opts.SetIncludeVector(false); err != nil {
		t.Fatalf("SetIncludeVector() failed: %v", err)
	}

	iter, err := collection.CreateIterator(opts)
	if err != nil {
		t.Fatalf("CreateIterator() failed: %v", err)
	}

	docs := iteratorAllDocs(t, iter)
	iter.Close()
	defer FreeDocs(docs)

	if len(docs) != 2 {
		t.Fatalf("iterator returned %d docs, want 2", len(docs))
	}
	for _, doc := range docs {
		if doc.GetPK() == "" {
			t.Error("iterator returned doc with empty PK")
		}
		if doc.HasField("id") {
			t.Errorf("doc %s unexpectedly contains scalar field id", doc.GetPK())
		}
	}
}

func TestDocIteratorSnapshotIsolation(t *testing.T) {
	const docCount = 4
	collection := newIteratorTestCollection(t, docCount)

	iter, err := collection.CreateIterator(nil)
	if err != nil {
		t.Fatalf("CreateIterator() failed: %v", err)
	}

	// Writes after iterator creation must not be visible to the iterator.
	extraDoc := createTestDoc("doc_late", "late", []float32{0.9, 0.9, 0.9, 0.9})
	if _, err := collection.Insert([]*Doc{extraDoc}); err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}
	extraDoc.Destroy()

	docs := iteratorAllDocs(t, iter)
	iter.Close()
	defer FreeDocs(docs)

	if len(docs) != docCount {
		pks := make([]string, 0, len(docs))
		for _, doc := range docs {
			pks = append(pks, doc.GetPK())
		}
		sort.Strings(pks)
		t.Fatalf("iterator returned %d docs (%v), want %d (snapshot isolation)",
			len(docs), pks, docCount)
	}
	for _, doc := range docs {
		if doc.GetPK() == "doc_late" {
			t.Fatal("iterator observed a document written after its creation")
		}
	}
}

func TestDocIteratorCloseIdempotent(t *testing.T) {
	collection := newIteratorTestCollection(t, 1)

	iter, err := collection.CreateIterator(nil)
	if err != nil {
		t.Fatalf("CreateIterator() failed: %v", err)
	}

	iter.Close()
	iter.Close()

	if _, err := iter.Next(); err != io.EOF {
		t.Errorf("Next() after Close() = %v, want io.EOF", err)
	}
}

func TestIteratorOptionsDestroyIdempotent(t *testing.T) {
	opts := NewIteratorOptions()
	if opts == nil {
		t.Fatal("NewIteratorOptions returned nil")
	}
	opts.Destroy()
	opts.Destroy()
}
