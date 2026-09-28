package zvec

/*
#include "zvec/c_api.h"
#include <stdlib.h>
#include <string.h>

// Shared status type (see errors.go for the rationale): error code plus the
// thread-local last-error message fetched in the SAME native call. The Go
// zvec API passes NEVER passes Go-pointer out-params to the wrappers below —
// every Go pointer in a cgo argument list forces a heap allocation of the
// call's argument frame — so all outputs come back by value.
typedef struct zvec_go_status {
  zvec_error_code_t code;
  char *err_msg; // malloc'd copy when non-NULL; free with zvec_free
} zvec_go_status_t;

// Field value reference (string / vector / binary): pointer into document
// memory, valid while the owning Doc lives.
typedef struct zvec_go_value {
  zvec_go_status_t status;
  const void *value;
  size_t value_size;
} zvec_go_value_t;

// Fixed-width scalar payload for the basic field types (all ≤ 8 bytes).
typedef struct zvec_go_scalar {
  zvec_go_status_t status;
  unsigned char value[8];
} zvec_go_scalar_t;

// Field-name list result.
typedef struct zvec_go_names {
  zvec_go_status_t status;
  char **names;
  size_t count;
} zvec_go_names_t;

static zvec_go_status_t zvec_go_doc_add_field_by_value_ex(
    zvec_doc_t *doc, const char *field_name, zvec_data_type_t type,
    void *value, size_t value_size) {
  zvec_go_status_t s;
  s.code = zvec_doc_add_field_by_value(doc, field_name, type, value, value_size);
  s.err_msg = NULL;
  if (s.code != ZVEC_OK) zvec_get_last_error(&s.err_msg);
  return s;
}
static zvec_go_status_t zvec_go_doc_set_field_null_ex(
    zvec_doc_t *doc, const char *field_name) {
  zvec_go_status_t s;
  s.code = zvec_doc_set_field_null(doc, field_name);
  s.err_msg = NULL;
  if (s.code != ZVEC_OK) zvec_get_last_error(&s.err_msg);
  return s;
}
static zvec_go_status_t zvec_go_doc_remove_field_ex(
    zvec_doc_t *doc, const char *field_name) {
  zvec_go_status_t s;
  s.code = zvec_doc_remove_field(doc, field_name);
  s.err_msg = NULL;
  if (s.code != ZVEC_OK) zvec_get_last_error(&s.err_msg);
  return s;
}
static zvec_go_value_t zvec_go_doc_get_field_value_pointer_ex(
    zvec_doc_t *doc, const char *field_name, zvec_data_type_t type) {
  zvec_go_value_t r;
  r.value = NULL;
  r.value_size = 0;
  r.status.code = zvec_doc_get_field_value_pointer(
      doc, field_name, type, &r.value, &r.value_size);
  r.status.err_msg = NULL;
  if (r.status.code != ZVEC_OK) zvec_get_last_error(&r.status.err_msg);
  return r;
}
static zvec_go_scalar_t zvec_go_doc_get_field_value_basic_ex(
    zvec_doc_t *doc, const char *field_name, zvec_data_type_t type,
    size_t value_size) {
  zvec_go_scalar_t r;
  memset(r.value, 0, sizeof(r.value));
  r.status.code = zvec_doc_get_field_value_basic(
      doc, field_name, type, (void *)r.value, value_size);
  r.status.err_msg = NULL;
  if (r.status.code != ZVEC_OK) zvec_get_last_error(&r.status.err_msg);
  return r;
}
static zvec_go_names_t zvec_go_doc_get_field_names_ex(zvec_doc_t *doc) {
  zvec_go_names_t r;
  r.names = NULL;
  r.count = 0;
  r.status.code = zvec_doc_get_field_names(doc, &r.names, &r.count);
  r.status.err_msg = NULL;
  if (r.status.code != ZVEC_OK) zvec_get_last_error(&r.status.err_msg);
  return r;
}
*/
import "C"
import (
	"sync"
	"sync/atomic"
	"unsafe"
)

// maxInternedFieldNames bounds the intern table so a hostile or dynamic
// field-name vocabulary cannot grow it unboundedly. Once the cap is reached,
// cFieldName degrades to a fresh C.CString per call.
const maxInternedFieldNames = 1024

// internedFieldNames caches *C.char copies of schema field names. Entries
// are never freed: they are C memory referenced outside any Go GC reach, and
// the vocabulary (schema field names) is small and long-lived. Only call
// cFieldName with field names — never with primary keys, field values, or
// filter expressions, which are unbounded.
var internedFieldNames sync.Map // string → *C.char

var internedFieldCount atomic.Int32

// cFieldName returns a NUL-terminated C copy of a schema field name. Repeated
// names hit the intern cache and cost a single sync.Map load (~10ns) instead
// of a native malloc+memcpy+free pair plus two cgo transitions, which every
// per-field accessor would otherwise pay on each call. The returned release
// function is nil for interned (never-freed) entries and must be deferred by
// the caller otherwise.
func cFieldName(name string) (*C.char, func()) {
	if cached, ok := internedFieldNames.Load(name); ok {
		return cached.(*C.char), nil
	}
	if internedFieldCount.Add(1) > maxInternedFieldNames {
		internedFieldCount.Add(-1)
		cName := C.CString(name)
		return cName, func() { C.free(unsafe.Pointer(cName)) }
	}
	cName := C.CString(name)
	if actual, loaded := internedFieldNames.LoadOrStore(name, cName); loaded {
		// Another goroutine interned the same name first; keep theirs.
		internedFieldCount.Add(-1)
		C.free(unsafe.Pointer(cName))
		return actual.(*C.char), nil
	}
	return cName, nil
}

// Doc represents a document in the zvec vector database.
// It wraps the C zvec_doc_t handle and manages ownership.
type Doc struct {
	handle *C.zvec_doc_t
	owned  bool
}

// NewDoc creates a new document with owned=true.
// The caller is responsible for calling Destroy() when done.
func NewDoc() *Doc {
	handle := C.zvec_doc_create()
	if handle == nil {
		return nil
	}
	return &Doc{handle: handle, owned: true}
}

// Destroy releases the document resources.
// Only effective when owned=true.
//
// Destroy is safe to call concurrently or repeatedly (including via FreeDocs
// from multiple goroutines): the native handle is atomically claimed, so the
// underlying document is released exactly once.
func (d *Doc) Destroy() {
	if d == nil || !d.owned {
		return
	}
	// handle is the first field of Doc, so &d.handle is pointer-aligned and
	// can be swapped atomically. SWAP-and-check claims the handle exactly
	// once across racing Destroy calls.
	h := (*unsafe.Pointer)(unsafe.Pointer(&d.handle))
	if old := atomic.SwapPointer(h, nil); old != nil {
		C.zvec_doc_destroy((*C.zvec_doc_t)(old))
	}
}

// Clear clears all fields and metadata from the document.
func (d *Doc) Clear() {
	C.zvec_doc_clear(d.handle)
}

// SetPK sets the primary key of the document.
// SetPK keeps its C.CString (rather than an interned copy): the native API
// takes a NUL-terminated const char*, and primary keys are unbounded
// per-document values that must never enter the bounded field-name intern
// table.
func (d *Doc) SetPK(pk string) {
	cPK := C.CString(pk)
	defer C.free(unsafe.Pointer(cPK))
	C.zvec_doc_set_pk(d.handle, cPK)
}

// SetDocID sets the document ID.
func (d *Doc) SetDocID(docID uint64) {
	C.zvec_doc_set_doc_id(d.handle, C.uint64_t(docID))
}

// SetScore sets the document score.
func (d *Doc) SetScore(score float32) {
	C.zvec_doc_set_score(d.handle, C.float(score))
}

// SetOperator sets the document operator.
func (d *Doc) SetOperator(op DocOperator) {
	C.zvec_doc_set_operator(d.handle, C.zvec_doc_operator_t(op))
}

// AddStringField adds a string field to the document.
func (d *Doc) AddStringField(name, value string) error {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	cValue := C.CString(value)
	defer C.free(unsafe.Pointer(cValue))
	return statusError(C.zvec_go_doc_add_field_by_value_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_STRING,
		unsafe.Pointer(cValue), C.size_t(len(value)),
	))
}

// AddBoolField adds a boolean field to the document.
func (d *Doc) AddBoolField(name string, value bool) error {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	var cValue C.bool = C.bool(value)
	return statusError(C.zvec_go_doc_add_field_by_value_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_BOOL,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddInt32Field adds an int32 field to the document.
func (d *Doc) AddInt32Field(name string, value int32) error {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	cValue := C.int32_t(value)
	return statusError(C.zvec_go_doc_add_field_by_value_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_INT32,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddInt64Field adds an int64 field to the document.
func (d *Doc) AddInt64Field(name string, value int64) error {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	cValue := C.int64_t(value)
	return statusError(C.zvec_go_doc_add_field_by_value_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_INT64,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddUint32Field adds a uint32 field to the document.
func (d *Doc) AddUint32Field(name string, value uint32) error {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	cValue := C.uint32_t(value)
	return statusError(C.zvec_go_doc_add_field_by_value_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_UINT32,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddUint64Field adds a uint64 field to the document.
func (d *Doc) AddUint64Field(name string, value uint64) error {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	cValue := C.uint64_t(value)
	return statusError(C.zvec_go_doc_add_field_by_value_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_UINT64,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddFloatField adds a float32 field to the document.
func (d *Doc) AddFloatField(name string, value float32) error {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	cValue := C.float(value)
	return statusError(C.zvec_go_doc_add_field_by_value_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_FLOAT,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddDoubleField adds a float64 field to the document.
func (d *Doc) AddDoubleField(name string, value float64) error {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	cValue := C.double(value)
	return statusError(C.zvec_go_doc_add_field_by_value_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_DOUBLE,
		unsafe.Pointer(&cValue), C.size_t(unsafe.Sizeof(cValue)),
	))
}

// AddVectorFP32Field adds a float32 vector field to the document.
func (d *Doc) AddVectorFP32Field(name string, vector []float32) error {
	if len(vector) == 0 {
		return &Error{Code: ErrInvalidArgument, Message: "vector cannot be empty"}
	}
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	return statusError(C.zvec_go_doc_add_field_by_value_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_VECTOR_FP32,
		unsafe.Pointer(&vector[0]), C.size_t(len(vector)*4),
	))
}

// AddBinaryField adds a binary field to the document.
// The data must not be empty; use SetFieldNull for null values.
func (d *Doc) AddBinaryField(name string, data []byte) error {
	if len(data) == 0 {
		return &Error{Code: ErrInvalidArgument, Message: "binary data cannot be empty"}
	}
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	return statusError(C.zvec_go_doc_add_field_by_value_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_BINARY,
		unsafe.Pointer(&data[0]), C.size_t(len(data)),
	))
}

// SetFieldNull sets a field to null.
func (d *Doc) SetFieldNull(name string) error {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	return statusError(C.zvec_go_doc_set_field_null_ex(d.handle, cName))
}

// RemoveField removes a field from the document.
func (d *Doc) RemoveField(name string) error {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	return statusError(C.zvec_go_doc_remove_field_ex(d.handle, cName))
}

// GetPK returns the primary key of the document.
// Uses the zero-copy pointer getter: the string lives in native doc memory
// for as long as the Doc exists (the same lifetime GetStringField relies
// on), avoiding a native malloc+strcpy+free round trip per result doc.
// C.GoString copies immediately, so the returned Go string never references
// native memory.
func (d *Doc) GetPK() string {
	cPK := C.zvec_doc_get_pk_pointer(d.handle)
	if cPK == nil {
		return ""
	}
	return C.GoString(cPK)
}

// GetDocID returns the document ID.
func (d *Doc) GetDocID() uint64 {
	return uint64(C.zvec_doc_get_doc_id(d.handle))
}

// GetScore returns the document score.
func (d *Doc) GetScore() float32 {
	return float32(C.zvec_doc_get_score(d.handle))
}

// GetOperator returns the document operator.
func (d *Doc) GetOperator() DocOperator {
	return DocOperator(C.zvec_doc_get_operator(d.handle))
}

// GetFieldCount returns the number of fields in the document.
func (d *Doc) GetFieldCount() int {
	return int(C.zvec_doc_get_field_count(d.handle))
}

// IsEmpty returns true if the document has no fields.
func (d *Doc) IsEmpty() bool {
	return bool(C.zvec_doc_is_empty(d.handle))
}

// GetStringField returns the string value of a field.
func (d *Doc) GetStringField(name string) (string, error) {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	result := C.zvec_go_doc_get_field_value_pointer_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_STRING,
	)
	if err := statusError(result.status); err != nil {
		return "", err
	}
	return C.GoStringN((*C.char)(result.value), C.int(result.value_size)), nil
}

// GetBoolField returns the boolean value of a field.
func (d *Doc) GetBoolField(name string) (bool, error) {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	var value C.bool
	result := C.zvec_go_doc_get_field_value_basic_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_BOOL,
		C.size_t(unsafe.Sizeof(value)),
	)
	if err := statusError(result.status); err != nil {
		return false, err
	}
	value = *(*C.bool)(unsafe.Pointer(&result.value[0]))
	return bool(value), nil
}

// GetInt32Field returns the int32 value of a field.
func (d *Doc) GetInt32Field(name string) (int32, error) {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	var value C.int32_t
	result := C.zvec_go_doc_get_field_value_basic_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_INT32,
		C.size_t(unsafe.Sizeof(value)),
	)
	if err := statusError(result.status); err != nil {
		return 0, err
	}
	value = *(*C.int32_t)(unsafe.Pointer(&result.value[0]))
	return int32(value), nil
}

// GetInt64Field returns the int64 value of a field.
func (d *Doc) GetInt64Field(name string) (int64, error) {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	var value C.int64_t
	result := C.zvec_go_doc_get_field_value_basic_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_INT64,
		C.size_t(unsafe.Sizeof(value)),
	)
	if err := statusError(result.status); err != nil {
		return 0, err
	}
	value = *(*C.int64_t)(unsafe.Pointer(&result.value[0]))
	return int64(value), nil
}

// GetUint32Field returns the uint32 value of a field.
func (d *Doc) GetUint32Field(name string) (uint32, error) {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	var value C.uint32_t
	result := C.zvec_go_doc_get_field_value_basic_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_UINT32,
		C.size_t(unsafe.Sizeof(value)),
	)
	if err := statusError(result.status); err != nil {
		return 0, err
	}
	value = *(*C.uint32_t)(unsafe.Pointer(&result.value[0]))
	return uint32(value), nil
}

// GetUint64Field returns the uint64 value of a field.
func (d *Doc) GetUint64Field(name string) (uint64, error) {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	var value C.uint64_t
	result := C.zvec_go_doc_get_field_value_basic_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_UINT64,
		C.size_t(unsafe.Sizeof(value)),
	)
	if err := statusError(result.status); err != nil {
		return 0, err
	}
	value = *(*C.uint64_t)(unsafe.Pointer(&result.value[0]))
	return uint64(value), nil
}

// GetFloatField returns the float32 value of a field.
func (d *Doc) GetFloatField(name string) (float32, error) {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	var value C.float
	result := C.zvec_go_doc_get_field_value_basic_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_FLOAT,
		C.size_t(unsafe.Sizeof(value)),
	)
	if err := statusError(result.status); err != nil {
		return 0, err
	}
	value = *(*C.float)(unsafe.Pointer(&result.value[0]))
	return float32(value), nil
}

// GetDoubleField returns the float64 value of a field.
func (d *Doc) GetDoubleField(name string) (float64, error) {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	var value C.double
	result := C.zvec_go_doc_get_field_value_basic_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_DOUBLE,
		C.size_t(unsafe.Sizeof(value)),
	)
	if err := statusError(result.status); err != nil {
		return 0, err
	}
	value = *(*C.double)(unsafe.Pointer(&result.value[0]))
	return float64(value), nil
}

// GetVectorFP32Field returns the float32 vector value of a field.
func (d *Doc) GetVectorFP32Field(name string) ([]float32, error) {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	field := C.zvec_go_doc_get_field_value_pointer_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_VECTOR_FP32,
	)
	if err := statusError(field.status); err != nil {
		return nil, err
	}
	count := int(field.value_size) / 4
	result := make([]float32, count)
	copy(result, unsafe.Slice((*float32)(field.value), count))
	return result, nil
}

// GetVectorFP32FieldInto copies a vector into dst, growing it when capacity is insufficient.
// The returned data is independent of the document and remains valid after Destroy.
func (d *Doc) GetVectorFP32FieldInto(name string, dst []float32) ([]float32, error) {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	field := C.zvec_go_doc_get_field_value_pointer_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_VECTOR_FP32,
	)
	if err := statusError(field.status); err != nil {
		return nil, err
	}
	count := int(field.value_size) / 4
	if cap(dst) < count {
		dst = make([]float32, count)
	} else {
		dst = dst[:count]
	}
	copy(dst, unsafe.Slice((*float32)(field.value), count))
	return dst, nil
}

// GetBinaryField returns the binary value of a field as a fresh byte slice.
func (d *Doc) GetBinaryField(name string) ([]byte, error) {
	return d.GetBinaryFieldInto(name, nil)
}

// GetBinaryFieldInto copies a binary field into dst, growing it when
// capacity is insufficient, mirroring GetVectorFP32FieldInto. The value
// lives in native doc memory, so the Go copy here is mandatory for
// lifetime safety.
func (d *Doc) GetBinaryFieldInto(name string, dst []byte) ([]byte, error) {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	field := C.zvec_go_doc_get_field_value_pointer_ex(
		d.handle, cName, C.ZVEC_DATA_TYPE_BINARY,
	)
	if err := statusError(field.status); err != nil {
		return nil, err
	}
	count := int(field.value_size)
	if cap(dst) < count {
		dst = make([]byte, count)
	} else {
		dst = dst[:count]
	}
	copy(dst, unsafe.Slice((*byte)(field.value), count))
	return dst, nil
}

// HasField returns true if the document has a field with the given name.
func (d *Doc) HasField(name string) bool {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	return bool(C.zvec_doc_has_field(d.handle, cName))
}

// HasFieldValue returns true if the document has a field with the given name and a non-null value.
func (d *Doc) HasFieldValue(name string) bool {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	return bool(C.zvec_doc_has_field_value(d.handle, cName))
}

// IsFieldNull returns true if the field with the given name is null.
func (d *Doc) IsFieldNull(name string) bool {
	cName, release := cFieldName(name)
	if release != nil {
		defer release()
	}
	return bool(C.zvec_doc_is_field_null(d.handle, cName))
}

// GetFieldNames returns a list of all field names in the document.
func (d *Doc) GetFieldNames() ([]string, error) {
	result := C.zvec_go_doc_get_field_names_ex(d.handle)
	if err := statusError(result.status); err != nil {
		return nil, err
	}
	defer C.zvec_free_str_array(result.names, result.count)
	nameSlice := unsafe.Slice(result.names, int(result.count))
	names := make([]string, int(result.count))
	for i := 0; i < int(result.count); i++ {
		names[i] = C.GoString(nameSlice[i])
	}
	return names, nil
}
