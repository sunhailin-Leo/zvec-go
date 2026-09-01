//go:build cgo && !purego

package zvec

// loadedLibraryDirs returns the directories of loaded zvec C-API libraries.
// In cgo builds the library is resolved by the dynamic linker at process
// startup, so the path is not tracked; dict discovery falls back to the
// package source, working, and executable directories.
func loadedLibraryDirs() []string {
	return nil
}
