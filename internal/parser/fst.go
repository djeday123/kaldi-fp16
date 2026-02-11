package parser

import (
	"bufio"
	"encoding/binary"
	"io"
	"math"
)

const (
	kFstMagicNumber = 0x7eb2fdd6
	kNoStateId      = -1
)

var kPosInfinity = float32(math.Inf(1))

// ReadFst reads OpenFst binary format (compact_acceptor or vector)
func ReadFst(reader *bufio.Reader) *Fst {
	// Read header
	magic := readInt32Raw(reader)
	if magic != kFstMagicNumber {
		return nil
	}

	fstType := readString(reader)
	arcType := readString(reader)

	if arcType != "standard" {
		return nil
	}

	switch fstType {
	case "compact_acceptor":
		return readCompactAcceptor(reader)
	case "vector":
		return readVectorFst(reader)
	default:
		return nil
	}
}

// readFstHeader reads the common header fields after fstType/arcType
type fstHeader struct {
	version    int32
	flags      int32
	properties uint64
	start      int64
	numStates  int64
	numArcs    int64
}

func readFstHeaderFields(reader *bufio.Reader) fstHeader {
	return fstHeader{
		version:    readInt32Raw(reader),
		flags:      readInt32Raw(reader),
		properties: readUint64Raw(reader),
		start:      readInt64Raw(reader),
		numStates:  readInt64Raw(reader),
		numArcs:    readInt64Raw(reader),
	}
}

// readCompactAcceptor reads OpenFst CompactAcceptor format
func readCompactAcceptor(reader *bufio.Reader) *Fst {
	h := readFstHeaderFields(reader)

	fst := &Fst{
		Start:      h.start,
		NumStates:  h.numStates,
		NumArcs:    h.numArcs,
		Properties: h.properties,
		States:     make([]FstState, h.numStates),
	}

	// Read states offsets: (numStates + 1) × uint32
	statesOffsets := make([]uint32, h.numStates+1)
	for i := int64(0); i <= h.numStates; i++ {
		statesOffsets[i] = readUint32Raw(reader)
	}

	// ncompacts = states[numStates]
	ncompacts := statesOffsets[h.numStates]

	// Read compacts: Element = (label:int32, weight:float32, nextstate:int32)
	type compactElement struct {
		label     int32
		weight    float32
		nextState int32
	}
	compacts := make([]compactElement, ncompacts)
	for i := uint32(0); i < ncompacts; i++ {
		compacts[i].label = readInt32Raw(reader)
		compacts[i].weight = readFloat32Raw(reader)
		compacts[i].nextState = readInt32Raw(reader)
	}

	// Build states from compacts
	for s := int64(0); s < h.numStates; s++ {
		startIdx := statesOffsets[s]
		endIdx := statesOffsets[s+1]

		state := &fst.States[s]
		state.Final = kPosInfinity // default: not final

		for i := startIdx; i < endIdx; i++ {
			elem := compacts[i]

			if elem.nextState == kNoStateId {
				state.Final = elem.weight
			} else {
				state.Arcs = append(state.Arcs, FstArc{
					Label:     elem.label,
					Weight:    elem.weight,
					NextState: elem.nextState,
				})
			}
		}
	}

	return fst
}

// readVectorFst reads OpenFst VectorFst<StdArc> format
// Format per state: final_weight(float32), narcs(int64),
//
//	per arc: ilabel(int32), olabel(int32), weight(float32), nextstate(int32)
func readVectorFst(reader *bufio.Reader) *Fst {
	h := readFstHeaderFields(reader)

	fst := &Fst{
		Start:      h.start,
		NumStates:  h.numStates,
		NumArcs:    h.numArcs,
		Properties: h.properties,
		States:     make([]FstState, h.numStates),
	}

	for s := int64(0); s < h.numStates; s++ {
		state := &fst.States[s]

		// Final weight (Inf = not final)
		state.Final = readFloat32Raw(reader)

		// Number of arcs from this state
		narcs := readInt64Raw(reader)

		if narcs > 0 {
			state.Arcs = make([]FstArc, narcs)
			for a := int64(0); a < narcs; a++ {
				ilabel := readInt32Raw(reader)
				_ = readInt32Raw(reader) // olabel (== ilabel for acceptor)
				weight := readFloat32Raw(reader)
				nextState := readInt32Raw(reader)

				state.Arcs[a] = FstArc{
					Label:     ilabel,
					Weight:    weight,
					NextState: nextState,
				}
			}
		}
	}

	// Count actual arcs (header numArcs is 0 for vector FSTs)
	totalArcs := int64(0)
	for s := range fst.States {
		totalArcs += int64(len(fst.States[s].Arcs))
	}
	fst.NumArcs = totalArcs

	return fst
}

// ReadSupervision reads chain supervision block
func ReadSupervision(reader *bufio.Reader) *SupervisionBlock {
	sup := &SupervisionBlock{}

	for {
		b, err := reader.ReadByte()
		if err != nil {
			return sup
		}

		if b == '<' {
			tag, valid := tryReadTagFrom(reader)
			if !valid {
				continue
			}

			switch tag {
			case "Weight":
				reader.ReadByte() // space
				reader.ReadByte() // size (4)
				sup.Weight = readFloat32Raw(reader)

			case "NumSequences":
				reader.ReadByte() // space
				reader.ReadByte() // size (4)
				sup.NumSequences = int(readInt32Raw(reader))

			case "FramesPerSeq":
				reader.ReadByte() // space
				reader.ReadByte() // size (4)
				sup.FramesPerSeq = int(readInt32Raw(reader))

			case "LabelDim":
				reader.ReadByte() // space
				reader.ReadByte() // size (4)
				sup.LabelDim = int(readInt32Raw(reader))

			case "End2End":
				reader.ReadByte() // space
				e2e, _ := reader.ReadByte()
				sup.End2End = (e2e == 'T')
				if !sup.End2End {
					sup.Fst = ReadFst(reader)
				}

			case "/Supervision":
				return sup

			case "DW", "DW2":
				sup.DerivWeights = readDerivWeights(reader, tag)

			case "/NnetChainSup":
				return sup
			}
		}
	}
}

func readDerivWeights(reader *bufio.Reader, tag string) []float32 {
	reader.ReadByte() // space after tag

	if tag == "DW" {
		fv1, _ := reader.ReadByte()
		fv2, _ := reader.ReadByte()
		if fv1 != 'F' || fv2 != 'V' {
			return nil
		}
		reader.ReadByte() // space

		size := readInt32Raw(reader)
		weights := make([]float32, size)
		for i := int32(0); i < size; i++ {
			b, _ := reader.ReadByte()
			weights[i] = float32(b) / 255.0
		}
		return weights

	} else { // DW2
		fv1, _ := reader.ReadByte()
		fv2, _ := reader.ReadByte()
		if fv1 != 'F' || fv2 != 'V' {
			return nil
		}
		reader.ReadByte() // space
		reader.ReadByte() // size byte (4)

		size := readInt32Raw(reader)
		weights := make([]float32, size)
		for i := int32(0); i < size; i++ {
			weights[i] = readFloat32Raw(reader)
		}
		return weights
	}
}

// Helper functions for raw reading

func readInt32Raw(reader *bufio.Reader) int32 {
	var buf [4]byte
	io.ReadFull(reader, buf[:])
	return int32(binary.LittleEndian.Uint32(buf[:]))
}

func readUint32Raw(reader *bufio.Reader) uint32 {
	var buf [4]byte
	io.ReadFull(reader, buf[:])
	return binary.LittleEndian.Uint32(buf[:])
}

func readInt64Raw(reader *bufio.Reader) int64 {
	var buf [8]byte
	io.ReadFull(reader, buf[:])
	return int64(binary.LittleEndian.Uint64(buf[:]))
}

func readUint64Raw(reader *bufio.Reader) uint64 {
	var buf [8]byte
	io.ReadFull(reader, buf[:])
	return binary.LittleEndian.Uint64(buf[:])
}

func readFloat32Raw(reader *bufio.Reader) float32 {
	var buf [4]byte
	io.ReadFull(reader, buf[:])
	return math.Float32frombits(binary.LittleEndian.Uint32(buf[:]))
}

func readString(reader *bufio.Reader) string {
	var lenBuf [4]byte
	io.ReadFull(reader, lenBuf[:])
	length := binary.LittleEndian.Uint32(lenBuf[:])

	strBuf := make([]byte, length)
	io.ReadFull(reader, strBuf)
	return string(strBuf)
}

func tryReadTagFrom(reader *bufio.Reader) (string, bool) {
	var tagBytes []byte
	for {
		b, err := reader.ReadByte()
		if err != nil {
			return "", false
		}
		if b == '>' {
			break
		}
		if b == ' ' {
			reader.UnreadByte()
			break
		}
		if !isLetter(b) && !isDigit(b) && b != '/' && b != '_' {
			return "", false
		}
		tagBytes = append(tagBytes, b)
		if len(tagBytes) > 30 {
			return "", false
		}
	}
	if len(tagBytes) < 1 {
		return "", false
	}
	return string(tagBytes), true
}
