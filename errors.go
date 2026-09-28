package zvec

/*
#include "zvec/c_api.h"
#include <stdlib.h>

// zvec_go_status_t bundles an operation's error code with the thread-local
// last-error message fetched in the SAME native call (see the zvec_go_*_ex
// wrappers in doc.go/collection.go). Returning the pair by value means the
// Go side passes no Go-pointer out-params to cgo — every Go pointer in a
// cgo argument list forces a heap allocation of the call's argument frame,
// which used to cost 1-3 allocations per field accessor.
typedef struct zvec_go_status {
  zvec_error_code_t code;
  char *err_msg; // malloc'd copy when non-NULL; free with zvec_free (any thread)
} zvec_go_status_t;
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

// ErrorCode represents a zvec error code.
type ErrorCode int

const (
	ErrOK                 ErrorCode = 0
	ErrNotFound           ErrorCode = 1
	ErrAlreadyExists      ErrorCode = 2
	ErrInvalidArgument    ErrorCode = 3
	ErrPermissionDenied   ErrorCode = 4
	ErrFailedPrecondition ErrorCode = 5
	ErrResourceExhausted  ErrorCode = 6
	ErrUnavailable        ErrorCode = 7
	ErrInternalError      ErrorCode = 8
	ErrNotSupported       ErrorCode = 9
	ErrUnknown            ErrorCode = 10
)

// Error represents a zvec error with code and message.
type Error struct {
	Code    ErrorCode
	Message string
}

// ErrClosed is returned when an operation is attempted on a Collection whose
// Close or Destroy has already completed (or is completing concurrently).
var ErrClosed = &Error{Code: ErrFailedPrecondition, Message: "collection is closed"}

func (e *Error) Error() string {
	return fmt.Sprintf("zvec error %d: %s", e.Code, e.Message)
}

// Sentinel errors for common error codes.
var (
	ErrNotFoundError           = &Error{Code: ErrNotFound, Message: "resource not found"}
	ErrAlreadyExistsError      = &Error{Code: ErrAlreadyExists, Message: "resource already exists"}
	ErrInvalidArgumentError    = &Error{Code: ErrInvalidArgument, Message: "invalid argument"}
	ErrPermissionDeniedError   = &Error{Code: ErrPermissionDenied, Message: "permission denied"}
	ErrFailedPreconditionError = &Error{Code: ErrFailedPrecondition, Message: "failed precondition"}
	ErrResourceExhaustedError  = &Error{Code: ErrResourceExhausted, Message: "resource exhausted"}
	ErrUnavailableError        = &Error{Code: ErrUnavailable, Message: "unavailable"}
	ErrInternalErrorError      = &Error{Code: ErrInternalError, Message: "internal error"}
	ErrNotSupportedError       = &Error{Code: ErrNotSupported, Message: "not supported"}
	ErrUnknownError            = &Error{Code: ErrUnknown, Message: "unknown error"}
)

// IsNotFound checks if the error is a not found error.
func IsNotFound(err error) bool {
	var zvecErr *Error
	return errors.As(err, &zvecErr) && zvecErr.Code == ErrNotFound
}

// IsAlreadyExists checks if the error is an already exists error.
func IsAlreadyExists(err error) bool {
	var zvecErr *Error
	return errors.As(err, &zvecErr) && zvecErr.Code == ErrAlreadyExists
}

// IsInvalidArgument checks if the error is an invalid argument error.
func IsInvalidArgument(err error) bool {
	var zvecErr *Error
	return errors.As(err, &zvecErr) && zvecErr.Code == ErrInvalidArgument
}

// toError converts a C error code to a Go error.
// Returns nil if the error code is ZVEC_OK.
//
// Note: this helper fetches the native last-error message in a SECOND cgo
// transition after the failing operation already returned. zvec keeps the
// message in thread-local storage, and Go does not guarantee that a goroutine
// stays on the same OS thread across two separate cgo calls, so under
// concurrent failure load the message may be empty or belong to an unrelated
// call. Concurrent hot paths (collection DML/query/fetch/close and document
// field access) therefore use the zvec_go_*_ex preamble helpers, which fetch
// the message in the same native call as the operation, and convert with
// toErrorFromMsg instead. toError remains for construction-style and
// cold-path call sites that are not shared across goroutines.
func toError(code C.zvec_error_code_t) error {
	if code == C.ZVEC_OK {
		return nil
	}

	var cMsg *C.char
	C.zvec_get_last_error(&cMsg)
	defer func() {
		if cMsg != nil {
			C.zvec_free(unsafe.Pointer(cMsg))
		}
	}()

	message := "unknown error"
	if cMsg != nil {
		message = C.GoString(cMsg)
	}

	return &Error{
		Code:    ErrorCode(code),
		Message: message,
	}
}

// statusError converts a zvec_go_status_t (error code + last-error message
// captured in the same cgo transition by the zvec_go_*_ex wrappers) into a
// Go error and frees the message buffer. Freeing a malloc'd copy is
// thread-agnostic, so the freeing transition does not need to run on the
// thread that saw the failure. A nil message falls back to a generic text,
// mirroring toError.
func statusError(status C.zvec_go_status_t) error {
	if status.code == C.ZVEC_OK {
		return nil
	}

	defer func() {
		if status.err_msg != nil {
			C.zvec_free(unsafe.Pointer(status.err_msg))
		}
	}()

	message := "unknown error"
	if status.err_msg != nil {
		message = C.GoString(status.err_msg)
	}

	return &Error{
		Code:    ErrorCode(status.code),
		Message: message,
	}
}

// errorCodeToString converts an error code to its string representation.
func errorCodeToString(code ErrorCode) string {
	cStr := C.zvec_error_code_to_string(C.zvec_error_code_t(code))
	return C.GoString(cStr)
}
