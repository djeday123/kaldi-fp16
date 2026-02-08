package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"kaldi-fp16/internal/compare"
	"kaldi-fp16/internal/parser"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "analyze":
		cmdAnalyze(os.Args[2:])
	case "all":
		cmdAll(os.Args[2:])
	case "compare":
		cmdCompare(os.Args[2:])
	case "dump":
		cmdDump(os.Args[2:])
	case "verify":
		cmdVerify(os.Args[2:])
	case "help":
		printUsage()
	default:
		// Если передан файл напрямую - analyze
		if _, err := os.Stat(os.Args[1]); err == nil {
			cmdAnalyze(os.Args[1:])
		} else {
			fmt.Printf("Unknown command: %s\n", os.Args[1])
			printUsage()
			os.Exit(1)
		}
	}
}

func printUsage() {
	fmt.Println(`egstools - Kaldi EGS file analyzer

Commands:
  analyze <file>     Analyze single ark file
  all <dir>          Analyze all cegs.*.ark files
  compare <file>     Compare with Kaldi nnet3-chain-copy-egs
  dump <file>        Dump matrix data for verification
  help               Show this help
  verify <file>      Full verification against Kaldi (all matrix values)

Options:
  analyze: -n NUM    Number of examples to show (default: 5)
           -d LEVEL  Debug level 0-3 (default: 0)
  all:     -q        Quiet mode
  compare: -n NUM    Number of examples to compare (default: 3)
  dump:    -n NUM    Example number (default: 1)
           -r ROW    Row to dump, -1 for all (default: 0)
           -i INPUT  Input index 0=input, 1=ivector (default: 0)
           -v        Verbose output (detailed format)
  verify:  -n NUM    Number of examples to verify (default: 10)
           -t TOL    Tolerance for comparison (default: 0.001)

Examples:
  egstools analyze -n 10 cegs.1.ark
  egstools analyze -d 2 -n 110 cegs.1.ark
  egstools compare -n 5 cegs.1.ark
  egstools verify -n 10 -t 0.001 cegs.1.ark
  egstools all /data/kaldi-data/exp/chain/egs/`)
}

func cmdAnalyze(args []string) {
	fs := flag.NewFlagSet("analyze", flag.ExitOnError)
	num := fs.Int("n", 5, "Number of examples to show")
	debug := fs.Int("d", 0, "Debug level (0-3)")
	fs.Parse(args)

	parser.DebugLevel = *debug

	if fs.NArg() < 1 {
		fmt.Println("Usage: egstools analyze [-n NUM] [-d LEVEL] <file>")
		os.Exit(1)
	}

	filename := fs.Arg(0)
	reader, err := parser.NewReader(filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer reader.Close()

	count, valid, invalid := 0, 0, 0
	zeroWeight, badLabel := 0, 0
	frameSizes := make(map[int]int)

	for {
		ex, err := reader.ReadExample()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			break
		}
		if ex == nil {
			break
		}
		count++

		if ex.Validate() {
			valid++
			frameSizes[ex.Inputs[0].Matrix.Rows]++
			if ex.Supervision.Weight == 0 {
				zeroWeight++
			}
			if ex.Supervision.LabelDim != 3080 {
				badLabel++
			}
		} else {
			invalid++
		}

		if count <= *num {
			printExample(ex, count)
		}
	}

	fmt.Printf("\n=== %s ===\n", filename)
	fmt.Printf("Total: %d | Valid: %d | Invalid: %d\n", count, valid, invalid)
	fmt.Printf("Weight=0: %d | BadLabel: %d | Usable: %d\n", zeroWeight, badLabel, valid-zeroWeight-badLabel)
	fmt.Printf("\nFrame sizes: ")
	for size, cnt := range frameSizes {
		fmt.Printf("%d:%d ", size, cnt)
	}
	fmt.Println()
}

func cmdAll(args []string) {
	fs := flag.NewFlagSet("all", flag.ExitOnError)
	quiet := fs.Bool("q", false, "Quiet mode")
	fs.Parse(args)

	dir := "."
	if fs.NArg() > 0 {
		dir = fs.Arg(0)
	}

	files, _ := filepath.Glob(filepath.Join(dir, "cegs.*.ark"))
	if len(files) == 0 {
		fmt.Printf("No cegs.*.ark files found in %s\n", dir)
		os.Exit(1)
	}
	sort.Strings(files)

	fmt.Printf("Found %d files\n\n", len(files))

	totalExamples, totalValid, totalUsable := 0, 0, 0

	for _, f := range files {
		stats := analyzeFile(f)
		totalExamples += stats.Total
		totalValid += stats.Valid
		totalUsable += stats.Usable

		if !*quiet {
			fmt.Printf("%s: %d total, %d valid, %d usable\n",
				filepath.Base(f), stats.Total, stats.Valid, stats.Usable)
		}
	}

	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf("TOTAL: %d examples, %d valid, %d usable (%.1f%%)\n",
		totalExamples, totalValid, totalUsable,
		float64(totalUsable)/float64(totalExamples)*100)
}

func cmdCompare(args []string) {
	fs := flag.NewFlagSet("compare", flag.ExitOnError)
	num := fs.Int("n", 3, "Number of examples")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Println("Usage: egstools compare [-n NUM] <file>")
		os.Exit(1)
	}

	absPath, _ := filepath.Abs(fs.Arg(0))
	result, err := compare.CompareFiles(absPath, *num)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n%s\n", strings.Repeat("=", 40))
	fmt.Printf("Compared: %d | Matches: %d | Mismatches: %d\n",
		result.Compared, result.Matches, result.Mismatches)

	if result.Mismatches == 0 {
		fmt.Println("✅ Parser output matches Kaldi!")
	} else {
		fmt.Println("❌ Differences found")
	}
}

func analyzeFile(path string) *parser.FileStats {
	stats := &parser.FileStats{Path: path, FrameSizes: make(map[int]int)}

	reader, err := parser.NewReader(path)
	if err != nil {
		return stats
	}
	defer reader.Close()

	for {
		ex, _ := reader.ReadExample()
		if ex == nil {
			break
		}
		stats.Total++

		if ex.Validate() {
			stats.Valid++
			if ex.IsUsable() {
				stats.Usable++
			}
			if ex.Supervision.Weight == 0 {
				stats.ZeroWeight++
			}
			if ex.Supervision.LabelDim != 3080 {
				stats.UnusualLabelDim++
			}
		} else {
			stats.Invalid++
		}
	}
	return stats
}

func printExample(ex *parser.Example, num int) {
	fmt.Printf("\n[%d] %s\n", num, ex.Key)
	if len(ex.Inputs) > 0 {
		fmt.Printf("  Input: %s [%d x %d] %s\n",
			ex.Inputs[0].Name, ex.Inputs[0].Matrix.Rows, ex.Inputs[0].Matrix.Cols, ex.Inputs[0].Matrix.Type)
	}
	if len(ex.Inputs) > 1 {
		fmt.Printf("  Ivector: [%d x %d] %s\n",
			ex.Inputs[1].Matrix.Rows, ex.Inputs[1].Matrix.Cols, ex.Inputs[1].Matrix.Type)
	}
	fmt.Printf("  Supervision: weight=%.2f frames=%d labels=%d\n",
		ex.Supervision.Weight, ex.Supervision.FramesPerSeq, ex.Supervision.LabelDim)
}

func cmdDump(args []string) {
	fs := flag.NewFlagSet("dump", flag.ExitOnError)
	num := fs.Int("n", 1, "Example number to dump")
	row := fs.Int("r", 0, "Row to dump (-1 for all)")
	inputIdx := fs.Int("i", 0, "Input index (0=input, 1=ivector)")
	verbose := fs.Bool("v", false, "Verbose output (our detailed format)")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Println("Usage: egstools dump [-n NUM] [-r ROW] [-i INPUT] [-v] <file>")
		os.Exit(1)
	}

	reader, err := parser.NewReader(fs.Arg(0))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer reader.Close()

	for i := 1; i <= *num; i++ {
		ex, err := reader.ReadExample()
		if err != nil || ex == nil {
			fmt.Fprintf(os.Stderr, "Error reading example %d\n", i)
			break
		}
		if i < *num {
			continue
		}

		if *inputIdx >= len(ex.Inputs) {
			fmt.Printf("Input index %d not found (only %d inputs)\n", *inputIdx, len(ex.Inputs))
			os.Exit(1)
		}

		inp := ex.Inputs[*inputIdx]
		mat := inp.Matrix

		if *verbose {
			// Our detailed format
			fmt.Printf("Example %d: %s\n", i, ex.Key)
			fmt.Printf("%s [%d x %d] %s\n", inp.Name, mat.Rows, mat.Cols, mat.Type)
			fmt.Printf("Data length: %d (expected %d)\n\n", len(mat.Data), mat.Rows*mat.Cols)

			if *row == -1 {
				for r := 0; r < mat.Rows; r++ {
					fmt.Printf("Row %3d:", r)
					start := r * mat.Cols
					for c := 0; c < mat.Cols; c++ {
						fmt.Printf(" %.5f", mat.Data[start+c])
					}
					fmt.Println()
				}
			} else if *row >= 0 && *row < mat.Rows {
				fmt.Printf("Row %d:\n", *row)
				start := *row * mat.Cols
				for c := 0; c < mat.Cols; c++ {
					fmt.Printf("  [%2d] %.7f\n", c, mat.Data[start+c])
				}
			} else {
				fmt.Printf("Row %d out of range (0-%d)\n", *row, mat.Rows-1)
			}
		} else {
			// Kaldi-compatible text format (default)
			fmt.Printf("%s  ", ex.Key)
			if *row == -1 {
				fmt.Printf("[\n")
				for r := 0; r < mat.Rows; r++ {
					fmt.Printf("  ")
					start := r * mat.Cols
					for c := 0; c < mat.Cols; c++ {
						fmt.Printf("%.7g ", mat.Data[start+c])
					}
					if r == mat.Rows-1 {
						fmt.Printf("]\n")
					} else {
						fmt.Printf("\n")
					}
				}
			} else if *row >= 0 && *row < mat.Rows {
				fmt.Printf("row %d:\n", *row)
				start := *row * mat.Cols
				for c := 0; c < mat.Cols; c++ {
					fmt.Printf("%.7g\n", mat.Data[start+c])
				}
			} else {
				fmt.Printf("Row %d out of range (0-%d)\n", *row, mat.Rows-1)
			}
		}
	}
}

func cmdVerify(args []string) {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	numExamples := fs.Int("n", 10, "Number of examples to verify")
	tolerance := fs.Float64("t", 0.001, "Tolerance for float comparison")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Println("Usage: egstools verify [-n NUM] [-t TOL] <file>")
		os.Exit(1)
	}

	filename := fs.Arg(0)
	absPath, _ := filepath.Abs(filename)

	fmt.Printf("Verifying %d examples with tolerance %.6f\n", *numExamples, *tolerance)
	fmt.Printf("File: %s\n\n", absPath)

	results, err := compare.VerifyFull(absPath, *numExamples, float32(*tolerance))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	passed, failed := 0, 0
	for _, vr := range results {
		if vr.ErrorCount == 0 {
			fmt.Printf("[%3d] %s: ✅ %d×%d = %d values, max_diff=%.2e\n",
				vr.ExampleNum, vr.Key, vr.Rows, vr.Cols, vr.TotalValues, vr.MaxDiff)
			passed++
		} else {
			fmt.Printf("[%3d] %s: ❌ %d errors in %d values\n",
				vr.ExampleNum, vr.Key, vr.ErrorCount, vr.TotalValues)
			fmt.Printf("      First error: row=%d col=%d ours=%.7g kaldi=%.7g diff=%.2e\n",
				vr.FirstErrorRow, vr.FirstErrorCol, vr.FirstErrorOurs, vr.FirstErrorKaldi,
				vr.FirstErrorOurs-vr.FirstErrorKaldi)
			failed++
		}
	}

	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf("Verified: %d | Passed: %d | Failed: %d\n", len(results), passed, failed)

	if failed == 0 {
		fmt.Println("✅ All examples match Kaldi output!")
	} else {
		fmt.Println("❌ Some examples have differences")
		os.Exit(1)
	}
}
