package loader

import (
	"fmt"
	"math/rand"
	"path/filepath"

	"kaldi-fp16/internal/parser"
)

// EgsIterator итерирует по нескольким ark файлам
// Поддерживает glob паттерны и shuffle
type EgsIterator struct {
	arkPaths []string
	current  int
	reader   *parser.Reader
	shuffle  bool
}

// NewEgsIterator создаёт итератор по ark файлам
// pattern — glob паттерн, например "/data/egs/cegs.*.ark"
func NewEgsIterator(pattern string, shuffle bool) (*EgsIterator, error) {
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("glob error: %w", err)
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("no files match pattern: %s", pattern)
	}

	it := &EgsIterator{
		arkPaths: matches,
		shuffle:  shuffle,
	}

	if shuffle {
		rand.Shuffle(len(it.arkPaths), func(i, j int) {
			it.arkPaths[i], it.arkPaths[j] = it.arkPaths[j], it.arkPaths[i]
		})
	}

	return it, nil
}

// NewEgsIteratorFromPaths создаёт итератор из списка путей
func NewEgsIteratorFromPaths(paths []string, shuffle bool) (*EgsIterator, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("empty paths list")
	}

	it := &EgsIterator{
		arkPaths: make([]string, len(paths)),
		shuffle:  shuffle,
	}
	copy(it.arkPaths, paths)

	if shuffle {
		rand.Shuffle(len(it.arkPaths), func(i, j int) {
			it.arkPaths[i], it.arkPaths[j] = it.arkPaths[j], it.arkPaths[i]
		})
	}

	return it, nil
}

// Next возвращает следующий пример
// Автоматически переходит к следующему файлу при EOF
// Возвращает nil, nil когда все файлы прочитаны
func (it *EgsIterator) Next() (*parser.Example, error) {
	for {
		// Открыть следующий файл если нужно
		if it.reader == nil {
			if it.current >= len(it.arkPaths) {
				return nil, nil // все файлы прочитаны
			}
			var err error
			it.reader, err = parser.NewReader(it.arkPaths[it.current])
			if err != nil {
				it.current++
				continue
			}
		}

		// Читаем пример
		ex, err := it.reader.ReadExample()
		if err != nil {
			it.reader.Close()
			it.reader = nil
			it.current++
			return nil, fmt.Errorf("error in %s: %w", it.arkPaths[it.current-1], err)
		}

		// EOF — переходим к следующему файлу
		if ex == nil {
			it.reader.Close()
			it.reader = nil
			it.current++
			continue
		}

		return ex, nil
	}
}

// Reset сбрасывает итератор на начало
// При shuffle — перемешивает порядок файлов заново
func (it *EgsIterator) Reset() {
	if it.reader != nil {
		it.reader.Close()
		it.reader = nil
	}
	it.current = 0

	if it.shuffle {
		rand.Shuffle(len(it.arkPaths), func(i, j int) {
			it.arkPaths[i], it.arkPaths[j] = it.arkPaths[j], it.arkPaths[i]
		})
	}
}

// Close закрывает итератор
func (it *EgsIterator) Close() error {
	if it.reader != nil {
		return it.reader.Close()
	}
	return nil
}

// NumFiles возвращает количество ark файлов
func (it *EgsIterator) NumFiles() int {
	return len(it.arkPaths)
}

// CurrentFile возвращает путь текущего файла
func (it *EgsIterator) CurrentFile() string {
	if it.current < len(it.arkPaths) {
		return it.arkPaths[it.current]
	}
	return ""
}

// Progress возвращает прогресс (файлы прочитаны / всего)
func (it *EgsIterator) Progress() (int, int) {
	return it.current, len(it.arkPaths)
}
