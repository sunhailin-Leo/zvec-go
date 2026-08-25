//go:build cgo && !purego

package zvec

/*
#include "zvec/c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"io"
	"unsafe"
)

// IteratorOptions controls document iterator behavior.
//
// Available since zvec v0.7.0 (c_api: zvec_iterator_options_t).
type IteratorOptions struct {
	handle *C.zvec_iterator_options_t
}

// NewIteratorOptions creates iterator options with default values
// (all output fields, include vectors).
func NewIteratorOptions() *IteratorOptions {
	handle := C.zvec_iterator_options_create()
	if handle == nil {
		return nil
	}
	return &IteratorOptions{handle: handle}
}

// Destroy releases the iterator options resources.
func (o *IteratorOptions) Destroy() {
	if o.handle != nil {
		C.zvec_iterator_options_destroy(o.handle)
		o.handle = nil
	}
}

// SetOutputFields sets the scalar fields to return.
// A nil slice returns all fields; a non-nil empty slice returns no scalar
// fields (only the primary key / system columns).
func (o *IteratorOptions) SetOutputFields(fields []string) error {
	defer lockErrorThread()()
	if fields == nil {
		return toError(C.zvec_iterator_options_set_output_fields(o.handle, nil, 0))
	}
	cFields := make([]*C.char, len(fields))
	for i, f := range fields {
		cFields[i] = C.CString(f)
	}
	defer func() {
		for _, cf := range cFields {
			C.free(unsafe.Pointer(cf))
		}
	}()
	var ptr **C.char
	if len(cFields) > 0 {
		ptr = (**C.char)(unsafe.Pointer(&cFields[0]))
	} else {
		// Non-NULL pointer with count 0 means "no scalar fields".
		var dummy *C.char
		ptr = &dummy
	}
	return toError(C.zvec_iterator_options_set_output_fields(o.handle, ptr, C.size_t(len(cFields))))
}

// SetIncludeVector sets whether to include vector fields in the returned documents.
func (o *IteratorOptions) SetIncludeVector(include bool) error {
	defer lockErrorThread()()
	return toError(C.zvec_iterator_options_set_include_vector(o.handle, C.bool(include)))
}

// DocIterator iterates over all documents in a collection using an isolated
// snapshot taken at creation time (later writes are not visible).
//
// While an iterator is open, schema changes (create/drop index,
// add/alter/drop column) and destroy are rejected. Close every iterator
// before releasing the last collection handle.
//
// Available since zvec v0.7.0 (c_api: zvec_doc_iterator_t).
type DocIterator struct {
	handle *C.zvec_doc_iterator_t
}

// CreateIterator creates a document iterator over the collection.
// Pass nil for opts to use defaults (all fields, include vectors).
//
// Available since zvec v0.7.0 (c_api: zvec_collection_create_iterator).
func (c *Collection) CreateIterator(opts *IteratorOptions) (*DocIterator, error) {
	var cOpts *C.zvec_iterator_options_t
	if opts != nil {
		cOpts = opts.handle
	}
	var cIter *C.zvec_doc_iterator_t
	defer lockErrorThread()()
	if err := toError(C.zvec_collection_create_iterator(c.handle, cOpts, &cIter)); err != nil {
		return nil, err
	}
	return &DocIterator{handle: cIter}, nil
}

// Next advances the iterator and returns the next document.
// It returns io.EOF when iteration is complete.
// The caller is responsible for calling Destroy() on each returned Doc.
func (it *DocIterator) Next() (*Doc, error) {
	if it.handle == nil {
		return nil, io.EOF
	}
	var cDoc *C.zvec_doc_t
	defer lockErrorThread()()
	if err := toError(C.zvec_doc_iterator_next(it.handle, &cDoc)); err != nil {
		return nil, err
	}
	if cDoc == nil {
		return nil, io.EOF
	}
	return &Doc{handle: cDoc}, nil
}

// Close releases the iterator resources.
func (it *DocIterator) Close() {
	if it.handle != nil {
		C.zvec_doc_iterator_close(it.handle)
		it.handle = nil
	}
}
