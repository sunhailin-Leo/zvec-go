// This file exists to enable `go generate github.com/zvec-ai/zvec-go`
// for downloading pre-built zvec C-API libraries from GitHub Releases.
//
// Usage:
//
//	go generate github.com/zvec-ai/zvec-go
//
// This will run the download-libs command which:
// 1. Detects your current platform (darwin/arm64, linux/amd64, etc.)
// 2. Downloads the pre-built library archive from GitHub Releases
// 3. Extracts it to the ./lib directory
//
// The version is read from the VERSION file in the module root,
// or can be specified via the -version flag.
package zvec

//go:generate go run ./cmd/download-libs
