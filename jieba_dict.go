package zvec

import (
	"os"
	"path/filepath"
	"runtime"
)

// jiebaDictRequiredFiles lists the dictionary files that must be present for
// the `jieba` FTS tokenizer to work.
var jiebaDictRequiredFiles = []string{"jieba.dict.utf8", "hmm_model.utf8"}

// zvecDisableAutoJiebaDictEnv disables automatic jieba dict registration when
// set to a non-empty value.
const zvecDisableAutoJiebaDictEnv = "ZVEC_DISABLE_AUTO_JIEBA_DICT"

// FindJiebaDictDir locates a directory containing the jieba dictionary files
// (jieba.dict.utf8 and hmm_model.utf8). It probes, in order:
//
//  1. Directories next to the loaded zvec C-API library (purego builds),
//     covering both this repo's lib/<platform>/../data layout and the
//     upstream SDK lib/../data layout.
//  2. The zvec-go package source directory (vendored lib/data/jieba_dict or
//     a zvec submodule checkout).
//  3. The current working directory and the executable directory.
//
// It returns the first directory containing all required dictionary files,
// or an empty string when none is found.
func FindJiebaDictDir() string {
	for _, dir := range jiebaDictCandidateDirs() {
		if isJiebaDictDir(dir) {
			return dir
		}
	}
	return ""
}

// isJiebaDictDir reports whether dir contains all required jieba dict files.
func isJiebaDictDir(dir string) bool {
	if dir == "" {
		return false
	}
	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		return false
	}
	for _, name := range jiebaDictRequiredFiles {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			return false
		}
	}
	return true
}

// jiebaDictCandidateDirs builds the ordered list of directories to probe.
func jiebaDictCandidateDirs() []string {
	var candidates []string
	add := func(dir string) {
		if dir != "" {
			candidates = append(candidates, dir)
		}
	}

	// Next to the loaded C-API library (purego builds know the path).
	for _, libDir := range loadedLibraryDirs() {
		add(filepath.Join(libDir, "data", "jieba_dict"))
		// Upstream SDK layout: <sdk>/lib/libzvec_c_api.* + <sdk>/data/jieba_dict.
		// This repo's vendor layout: lib/<platform>/lib* + lib/data/jieba_dict.
		add(filepath.Join(libDir, "..", "data", "jieba_dict"))
	}

	// Relative to the zvec-go package source directory. This covers both the
	// vendored layout (lib/data/jieba_dict, populated by `go generate` or
	// scripts/package-libs.sh) and source builds against the zvec submodule.
	if _, file, _, ok := runtime.Caller(0); ok {
		pkgDir := filepath.Dir(file)
		add(filepath.Join(pkgDir, "lib", "data", "jieba_dict"))
		if matches, err := filepath.Glob(filepath.Join(pkgDir, "zvec", "thirdparty", "cppjieba", "*", "dict")); err == nil {
			for _, match := range matches {
				add(match)
			}
		}
	}

	// Relative to the working directory and the executable, mirroring the
	// purego library search paths.
	if cwd, err := os.Getwd(); err == nil {
		add(filepath.Join(cwd, "lib", "data", "jieba_dict"))
		add(filepath.Join(cwd, "data", "jieba_dict"))
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		add(filepath.Join(exeDir, "lib", "data", "jieba_dict"))
		add(filepath.Join(exeDir, "data", "jieba_dict"))
	}

	return candidates
}

// ensureDefaultJiebaDictDir auto-registers a process-wide default jieba dict
// directory when none is set yet. Called from Initialize. It is a no-op when
// a default is already registered, when the dictionaries cannot be found, or
// when ZVEC_DISABLE_AUTO_JIEBA_DICT is set. The C library gives the
// ZVEC_JIEBA_DICT_DIR env var and per-field jieba_dict_dir higher priority,
// so this default never overrides explicit user configuration.
func ensureDefaultJiebaDictDir() {
	if os.Getenv(zvecDisableAutoJiebaDictEnv) != "" {
		return
	}
	if GetDefaultJiebaDictDir() != "" {
		return
	}
	if dir := FindJiebaDictDir(); dir != "" {
		SetDefaultJiebaDictDir(dir)
	}
}
