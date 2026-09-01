//go:build purego || !cgo

package zvec

import "path/filepath"

// loadedLibraryDirs returns the directory of the zvec C-API library loaded by
// the purego backend, when one has been loaded.
func loadedLibraryDirs() []string {
	puregoLoadMu.Lock()
	defer puregoLoadMu.Unlock()
	if puregoLoadedLibPath == "" {
		return nil
	}
	return []string{filepath.Dir(puregoLoadedLibPath)}
}
