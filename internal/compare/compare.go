package compare

import (
	"fmt"
	"kaldi-fp16/internal/kaldi"
	"kaldi-fp16/internal/parser"
	"math"
)

// Result хранит результат сравнения
type Result struct {
	Compared   int
	Matches    int
	Mismatches int
}

// VerifyResult - детальный результат верификации одного примера
type VerifyResult struct {
	ExampleNum      int
	Key             string
	Rows            int
	Cols            int
	TotalValues     int
	ErrorCount      int
	MaxDiff         float32
	FirstErrorRow   int
	FirstErrorCol   int
	FirstErrorOurs  float32
	FirstErrorKaldi float32
}

// CompareFiles сравнивает наш парсер с Kaldi (структурное сравнение)
func CompareFiles(arkPath string, maxExamples int) (*Result, error) {
	fmt.Printf("Getting Kaldi text output for %d examples...\n", maxExamples)
	kaldiExamples, err := kaldi.GetTextOutput(arkPath, maxExamples)
	if err != nil {
		return nil, err
	}
	fmt.Printf("Kaldi returned %d examples\n", len(kaldiExamples))

	fmt.Println("Parsing with our parser...")
	reader, err := parser.NewReader(arkPath)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	var ourExamples []*parser.Example
	for len(ourExamples) < maxExamples {
		ex, err := reader.ReadExample()
		if err != nil {
			return nil, err
		}
		if ex == nil {
			break
		}
		ourExamples = append(ourExamples, ex)
	}
	fmt.Printf("Our parser returned %d examples\n", len(ourExamples))

	result := &Result{}
	minLen := len(ourExamples)
	if len(kaldiExamples) < minLen {
		minLen = len(kaldiExamples)
	}
	result.Compared = minLen

	for i := 0; i < minLen; i++ {
		fmt.Printf("\n--- Example %d ---\n", i+1)
		if compareOne(ourExamples[i], &kaldiExamples[i]) {
			result.Matches++
		} else {
			result.Mismatches++
		}
	}

	return result, nil
}

// VerifyFull выполняет полную верификацию всех значений матрицы
func VerifyFull(arkPath string, maxExamples int, tolerance float32) ([]VerifyResult, error) {
	kaldiExamples, err := kaldi.GetTextOutput(arkPath, maxExamples)
	if err != nil {
		return nil, err
	}

	reader, err := parser.NewReader(arkPath)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	var results []VerifyResult

	for i := 0; i < maxExamples; i++ {
		ex, err := reader.ReadExample()
		if err != nil || ex == nil {
			break
		}

		if i >= len(kaldiExamples) {
			break
		}

		vr := verifyExample(i+1, ex, &kaldiExamples[i], tolerance)
		results = append(results, vr)
	}

	return results, nil
}

func verifyExample(num int, ours *parser.Example, theirs *kaldi.TextExample, tolerance float32) VerifyResult {
	vr := VerifyResult{
		ExampleNum:    num,
		Key:           ours.Key,
		FirstErrorRow: -1,
		FirstErrorCol: -1,
	}

	if len(ours.Inputs) == 0 || len(theirs.AllRows) == 0 {
		return vr
	}

	mat := ours.Inputs[0].Matrix
	vr.Rows = mat.Rows
	vr.Cols = mat.Cols
	vr.TotalValues = mat.Rows * mat.Cols

	// Сравниваем по строкам
	maxRows := mat.Rows
	if len(theirs.AllRows) < maxRows {
		maxRows = len(theirs.AllRows)
	}

	for r := 0; r < maxRows; r++ {
		kaldiRow := theirs.AllRows[r]
		maxCols := mat.Cols
		if len(kaldiRow) < maxCols {
			maxCols = len(kaldiRow)
		}

		for c := 0; c < maxCols; c++ {
			oursVal := mat.Data[r*mat.Cols+c]
			kaldiVal := kaldiRow[c]

			diff := float32(math.Abs(float64(oursVal - kaldiVal)))

			if diff > vr.MaxDiff {
				vr.MaxDiff = diff
			}

			if diff > tolerance {
				vr.ErrorCount++
				if vr.FirstErrorRow == -1 {
					vr.FirstErrorRow = r
					vr.FirstErrorCol = c
					vr.FirstErrorOurs = oursVal
					vr.FirstErrorKaldi = kaldiVal
				}
			}
		}
	}

	return vr
}

func compareOne(ours *parser.Example, theirs *kaldi.TextExample) bool {
	allMatch := true

	fmt.Printf("Key: ours=%-40s kaldi=%-40s ", ours.Key, theirs.Key)
	if ours.Key == theirs.Key {
		fmt.Println("✅")
	} else {
		fmt.Println("(keys differ)")
	}

	fmt.Printf("NumInputs: ours=%d kaldi=%d ", ours.NumInputs, theirs.NumInputs)
	if ours.NumInputs == theirs.NumInputs {
		fmt.Println("✅")
	} else {
		fmt.Println("❌")
		allMatch = false
	}

	if len(ours.Inputs) > 0 {
		input := ours.Inputs[0]

		fmt.Printf("Input name: ours=%s kaldi=%s ", input.Name, theirs.InputName)
		if input.Name == theirs.InputName {
			fmt.Println("✅")
		} else {
			fmt.Println("❌")
			allMatch = false
		}

		fmt.Printf("Frames: ours=%d kaldi=%d ", input.Matrix.Rows, theirs.InputFrames)
		if input.Matrix.Rows == theirs.InputFrames {
			fmt.Println("✅")
		} else {
			fmt.Println("❌")
			allMatch = false
		}

		fmt.Printf("Dim: ours=%d kaldi=%d ", input.Matrix.Cols, theirs.InputDim)
		if input.Matrix.Cols == theirs.InputDim {
			fmt.Println("✅")
		} else {
			fmt.Println("❌")
			allMatch = false
		}

		if len(input.Matrix.FirstRow) >= 3 && len(theirs.FirstRow) >= 3 {
			fmt.Printf("First row[0:3]: ours=[%.2f, %.2f, %.2f] kaldi=[%.2f, %.2f, %.2f] ",
				input.Matrix.FirstRow[0], input.Matrix.FirstRow[1], input.Matrix.FirstRow[2],
				theirs.FirstRow[0], theirs.FirstRow[1], theirs.FirstRow[2])

			maxDiff := maxAbs(
				input.Matrix.FirstRow[0]-theirs.FirstRow[0],
				input.Matrix.FirstRow[1]-theirs.FirstRow[1],
				input.Matrix.FirstRow[2]-theirs.FirstRow[2],
			)
			if maxDiff < 0.5 {
				fmt.Printf("✅ (diff: %.4f)\n", maxDiff)
			} else {
				fmt.Printf("❌ (diff: %.4f)\n", maxDiff)
				allMatch = false
			}
		}
	}

	return allMatch
}

func maxAbs(vals ...float32) float32 {
	var m float32 = 0
	for _, v := range vals {
		a := float32(math.Abs(float64(v)))
		if a > m {
			m = a
		}
	}
	return m
}
