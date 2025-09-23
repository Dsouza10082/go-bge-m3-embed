package model

import (
	"log"
	"math"
	"os"
	"sort"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/sys/unix"
)

const modelPath = "./onnx/model.onnx"
const tokPath   = "./onnx/tokenizer.json"

type EmbeddingModel struct {
	Model  string `json:"model"`
	APIUrl string `json:"api_url"`
}

func NewEmbeddingModel() *EmbeddingModel {
	return &EmbeddingModel{}
}

func (e *EmbeddingModel) Cosine(a, b []float32) float64 {
	var dot float64
	for i := range a { dot += float64(a[i]) * float64(b[i]) }
	return dot // ambos já L2-normalizados (BGE-M3 costuma sair normalizado)
}

func (e *EmbeddingModel) l2norm(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	n := float32(math.Sqrt(s) + 1e-12)
	for i := range v {
		v[i] /= n
	}
}

func (e *EmbeddingModel) l2(a []float32) float64 {
	var s float64
	for _, x := range a { s += float64(x) * float64(x) }
	return math.Sqrt(s)
}

func (e *EmbeddingModel) meanPool(lastHidden []float32, seqLen, hidden int, attn []int64) []float32 {
	out := make([]float32, hidden)
	var cnt float32
	for t := 0; t < seqLen; t++ {
		if attn[t] == 0 {
			continue
		}
		row := t * hidden
		for h := 0; h < hidden; h++ {
			out[h] += lastHidden[row+h]
		}
		cnt++
	}
	if cnt > 0 {
		inv := 1.0 / cnt
		for h := range out {
			out[h] *= float32(inv)
		}
	}
	return out
}

func (e *EmbeddingModel) muteStderr(f func()) {
	// dup2(stderr -> backup)
	backup, _ := unix.Dup(int(os.Stderr.Fd()))
	devnull, _ := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	unix.Dup2(int(devnull.Fd()), int(os.Stderr.Fd()))
	f()
	// restaura stderr
	unix.Dup2(backup, int(os.Stderr.Fd()))
	unix.Close(backup)
	devnull.Close()
}

func (e *EmbeddingModel) NewTokenizer() *tokenizer.Tokenizer {
	tk, err := pretrained.FromFile(tokPath)
	if err != nil { log.Fatalf("tokenizer from file: %v", err) }

	tk.WithTruncation(&tokenizer.TruncationParams{ MaxLength: 1024 })
	return tk
}

type Scored struct{ Idx int; Score float64 }

func (e *EmbeddingModel) TopKCosine(db [][]float32, q []float32, k int) []Scored {

    best := make([]Scored, 0, k)
    push := func(s Scored) {
        if len(best) < k { best = append(best, s) } else {
            minI := 0
            for i := 1; i < k; i++ { if best[i].Score < best[minI].Score { minI = i } }
            if s.Score > best[minI].Score { best[minI] = s }
        }
    }
    for i := range db {
        var dot float64
        a := db[i]; for j := range a { dot += float64(a[j]) * float64(q[j]) }
        push(Scored{i, dot})
    }

    sort.Slice(best, func(i, j int) bool { return best[i].Score > best[j].Score })
    return best
}


func (e *EmbeddingModel) Embed(tk *tokenizer.Tokenizer, text string) []float32 {
	ort.SetSharedLibraryPath("./onnx/libonnxruntime.dylib")
	os.Unsetenv("DYLD_LIBRARY_PATH")
	
	e.muteStderr(func() {
		_ = ort.InitializeEnvironment()
	  })
	
	defer ort.DestroyEnvironment()

	en, err := tk.EncodeSingle(text)
	if err != nil { log.Fatalf("encode: %v", err) }
	idsInt, maskInt := en.GetIds(), en.GetAttentionMask()
	seq := len(idsInt)

	ids := make([]int64, seq)
	msk := make([]int64, seq)
	for i, v := range idsInt  { ids[i] = int64(v) }
	for i, v := range maskInt { msk[i] = int64(v) }

	inShape := ort.NewShape(1, int64(seq))
	tIDs,  err := ort.NewTensor[int64](inShape, ids);  if err != nil { log.Fatal(err) }
	tMask, err := ort.NewTensor[int64](inShape, msk);  if err != nil { log.Fatal(err) }
	defer tIDs.Destroy(); defer tMask.Destroy()

	outShape := ort.NewShape(1, 1024)
	tOut, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil { log.Fatal(err) }
	defer tOut.Destroy()

	// Cria a sessão já com os feeds/fetches e roda
	sess, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"sentence_embedding"}, // ajuste se seu ONNX usa "dense_vecs" ou "pooled_output"
		[]ort.Value{tIDs, tMask},
		[]ort.Value{tOut},
		nil,
	)
	if err != nil { log.Fatal(err) }
	defer sess.Destroy()

	if err := sess.Run(); err != nil { log.Fatal(err) }

	vec := tOut.GetData() // [1,1024] flatten
	// Normalmente já vem L2-normalizado; vamos só conferir a norma:
	_ = e.l2(vec)
	return vec 
}

func (e *EmbeddingModel) EmbedBGE3MText(text string) []float32 {

	ort.SetSharedLibraryPath("./onnx/libonnxruntime.dylib")
	os.Unsetenv("DYLD_LIBRARY_PATH")
	
	e.muteStderr(func() {
		_ = ort.InitializeEnvironment()
	  })
	
	defer ort.DestroyEnvironment()

	tk, err := pretrained.FromFile("./onnx/tokenizer.json")
	if err != nil {
		log.Fatal(err)
	}

	ps := tokenizer.NewPaddingStrategy(tokenizer.WithBatchLongest())
	tk.WithPadding(&tokenizer.PaddingParams{
		Strategy:  *ps,
		Direction: tokenizer.Right,
	})

	tk.WithTruncation(&tokenizer.TruncationParams{ MaxLength: 1024 })
	tk.WithPadding(&tokenizer.PaddingParams{Strategy: *ps})

	en, err := tk.EncodeSingle(text)
	if err != nil {
		log.Fatal(err)
	}

	idsInt := en.GetIds()            // []int
	maskInt := en.GetAttentionMask() // []int
	seqLen := len(idsInt)

	ids := make([]int64, seqLen)
	mask := make([]int64, seqLen)
	for i, v := range idsInt {
		ids[i] = int64(v)
	}
	for i, v := range maskInt {
		mask[i] = int64(v)
	}

	

	inpShape := ort.NewShape(1, int64(seqLen))
	tIDs, err := ort.NewTensor[int64](inpShape, ids)
	if err != nil {
		log.Fatal(err)
	}
	defer tIDs.Destroy()

	tMask, err := ort.NewTensor[int64](inpShape, mask)
	if err != nil {
		log.Fatal(err)
	}
	defer tMask.Destroy()



	outShape := ort.NewShape(1, 1024)
	tOut, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		log.Fatal(err)
	}
	defer tOut.Destroy()

	sess, _ := ort.NewAdvancedSession(
		"./onnx/model.onnx",
		[]string{"input_ids", "attention_mask"},
		[]string{"sentence_embedding"},  // ou "pooled_output"
		[]ort.Value{tIDs, tMask},
		[]ort.Value{tOut},
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}
	defer sess.Destroy()

	_ = sess.Run()
	emb := tOut.GetData()

	return emb

}

func (e *EmbeddingModel) Lo64(a []int) []int64 {
    r := make([]int64, len(a))
    for i, v := range a { r[i] = int64(v) }
    return r
}

func (e *EmbeddingModel) EmbedBatch(tk *tokenizer.Tokenizer, texts []string) [][]float32 {

	ort.SetSharedLibraryPath("./onnx/libonnxruntime.dylib")
	os.Unsetenv("DYLD_LIBRARY_PATH")
	
	e.muteStderr(func() {
		_ = ort.InitializeEnvironment()
	  })
	
	defer ort.DestroyEnvironment()

    encs := make([]*tokenizer.Encoding, len(texts))
    maxL := 0
    for i, t := range texts {
        e, err := tk.EncodeSingle(t); if err != nil { log.Fatal(err) }
        encs[i] = e
        if l := len(e.GetIds()); l > maxL { maxL = l }
    }
    B := len(texts)
    ids := make([]int64, B*maxL)
    mask:= make([]int64, B*maxL)
    for b := 0; b < B; b++ {
        en := encs[b]
        ii, mm := en.GetIds(), en.GetAttentionMask()
        copy(ids[b*maxL:b*maxL+len(ii)], e.Lo64(ii))
        copy(mask[b*maxL:b*maxL+len(mm)], e.Lo64(mm))
        // resto já fica 0 (padding id=0, mask=0) — ok para BGE
    }

    inShape := ort.NewShape(int64(B), int64(maxL))
    tIDs,  _ := ort.NewTensor[int64](inShape, ids)
    tMask, _ := ort.NewTensor[int64](inShape, mask)
    defer tIDs.Destroy(); defer tMask.Destroy()

    outShape := ort.NewShape(int64(B), 1024)
    tOut, _ := ort.NewEmptyTensor[float32](outShape)
    defer tOut.Destroy()

    sess, err := ort.NewAdvancedSession(
        modelPath,
        []string{"input_ids", "attention_mask"},
        []string{"sentence_embedding"},
        []ort.Value{tIDs, tMask},
        []ort.Value{tOut},
        nil,
    )
    if err != nil { log.Fatal(err) }
    defer sess.Destroy()

    if err := sess.Run(); err != nil { log.Fatal(err) }

    flat := tOut.GetData() // len = B*1024
    out := make([][]float32, B)
    for b := 0; b < B; b++ {
        out[b] = flat[b*1024 : (b+1)*1024]
        // normalmente já vem L2=~1.0; se quiser, cheque a norma aqui
    }
    return out
}


