//go:build integration

package zvec

import (
	"errors"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// TestConcurrentQueryErrorAttribution verifies the single-transition error
// capture (zvec_go_collection_query_ex): with goroutines issuing failing
// queries concurrently, every returned error must belong to the query that
// produced it. The native last-error lives in thread-local storage and Go
// does not pin a goroutine to an OS thread across cgo calls, so the previous
// pattern (a separate zvec_get_last_error transition inside toError) could
// return another goroutine's or another thread's message under concurrent
// failure load.
func TestConcurrentQueryErrorAttribution(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	collection, err := CreateAndOpen(t.TempDir()+"/concurrent_errors", schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	defer func() { _ = collection.Close() }()

	const goroutines = 8
	const rounds = 20
	for round := 0; round < rounds; round++ {
		errs := make([]error, goroutines)
		var wg sync.WaitGroup
		for g := 0; g < goroutines; g++ {
			wg.Add(1)
			go func(g int) {
				defer wg.Done()
				query := NewVectorQuery()
				if query == nil {
					errs[g] = errors.New("NewVectorQuery() returned nil")
					return
				}
				defer query.Destroy()
				marker := "missing_field_g" + strconv.Itoa(g)
				if err := query.SetFieldName(marker); err != nil {
					errs[g] = err
					return
				}
				if err := query.SetTopK(1); err != nil {
					errs[g] = err
					return
				}
				if err := query.SetQueryVector([]float32{0.1, 0.2, 0.3, 0.4}); err != nil {
					errs[g] = err
					return
				}
				_, errs[g] = collection.Query(query)
			}(g)
		}
		wg.Wait()

		for g, err := range errs {
			if err == nil {
				t.Fatalf("round %d goroutine %d: Query() unexpectedly succeeded", round, g)
			}
			marker := "missing_field_g" + strconv.Itoa(g)
			if !strings.Contains(err.Error(), marker) {
				t.Fatalf("round %d goroutine %d: cross-wired error message: %v", round, g, err)
			}
		}
	}
}

// TestCollectionCloseIdempotent verifies that concurrent Close calls perform
// exactly one teardown (no double free / SIGSEGV) and that operations after
// close consistently report ErrClosed.
func TestCollectionCloseIdempotent(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	collection, err := CreateAndOpen(t.TempDir()+"/close_idempotent", schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}

	const concurrency = 16
	var wg sync.WaitGroup
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := collection.Close(); err != nil {
				t.Errorf("concurrent Close() failed: %v", err)
			}
		}()
	}
	wg.Wait()

	if err := collection.Close(); err != nil {
		t.Errorf("Close() after close failed: %v", err)
	}
	if err := collection.Flush(); !errors.Is(err, ErrClosed) {
		t.Errorf("Flush() after Close() = %v, want ErrClosed", err)
	}
	doc := NewDoc()
	if doc == nil {
		t.Fatal("NewDoc() returned nil")
	}
	doc.Destroy()
	if _, err := collection.Insert([]*Doc{NewDoc()}); !errors.Is(err, ErrClosed) {
		t.Errorf("Insert() after Close() = %v, want ErrClosed", err)
	}
	if _, err := collection.Fetch([]string{"pk"}); !errors.Is(err, ErrClosed) {
		t.Errorf("Fetch() after Close() = %v, want ErrClosed", err)
	}
}

// TestCollectionDestroyIdempotent verifies that concurrent Destroy calls
// perform exactly one destroy+close teardown and that later Close/operations
// are consistent.
func TestCollectionDestroyIdempotent(t *testing.T) {
	schema := createTestSchema()
	defer schema.Destroy()

	collection, err := CreateAndOpen(t.TempDir()+"/destroy_idempotent", schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}

	const concurrency = 16
	var wg sync.WaitGroup
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = collection.Destroy()
		}()
	}
	wg.Wait()

	if err := collection.Close(); err != nil {
		t.Errorf("Close() after Destroy() failed: %v", err)
	}
	if err := collection.Flush(); !errors.Is(err, ErrClosed) {
		t.Errorf("Flush() after Destroy() = %v, want ErrClosed", err)
	}
}

// TestDocDestroyConcurrent verifies that concurrent and repeated Destroy /
// FreeDocs calls on the same documents release each native document exactly
// once (no double free). This is a smoke test: a regression would crash the
// process rather than fail an assertion.
func TestDocDestroyConcurrent(t *testing.T) {
	const docCount = 16
	docs := make([]*Doc, docCount)
	for i := range docs {
		docs[i] = NewDoc()
		if docs[i] == nil {
			t.Fatal("NewDoc() returned nil")
		}
		docs[i].SetPK("doc" + strconv.Itoa(i))
	}

	var wg sync.WaitGroup
	for repeat := 0; repeat < 4; repeat++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			FreeDocs(docs)
		}()
	}
	for i := range docs {
		wg.Add(1)
		go func(d *Doc) {
			defer wg.Done()
			d.Destroy()
		}(docs[i])
	}
	wg.Wait()

	// Repeated destroy after concurrent destroy must be a no-op.
	FreeDocs(docs)
}
