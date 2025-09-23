package golangbgem3embedder

// Golang-BGE-M3-embedder
//
// Author: Dsouza
// License: MIT

import (
	"fmt"

	co "github.com/Dsouza10082/ConcurrentOrderedMap"
	"github.com/Dsouza10082/go-bge-m3-embed/model"
)

const MEMORY_PATH = "./agent_memory/vecstore.json"

type GolangBGE3M3Embedder struct {
    EmbeddingModel *model.EmbeddingModel
    VecStore       *model.VecStore
    Verbose        bool
}

func NewGolangBGE3M3Embedder() *GolangBGE3M3Embedder {
    return &GolangBGE3M3Embedder{
        EmbeddingModel: model.NewEmbeddingModel(),
        VecStore:       model.NewVecStore(),
        Verbose:        false,
    }
}

func (e *GolangBGE3M3Embedder) Embed(text string) []float32 {
    tk := e.EmbeddingModel.NewTokenizer()
    vec := e.EmbeddingModel.Embed(tk, text)
    return vec
}

func (e *GolangBGE3M3Embedder) EmbedBatch(texts []string) [][]float32 {
    tk := e.EmbeddingModel.NewTokenizer()
    vecs := e.EmbeddingModel.EmbedBatch(tk, texts)
    return vecs
}

func (e *GolangBGE3M3Embedder) EmbedBGE3MText(text string) []float32 {
    return e.EmbeddingModel.EmbedBGE3MText(text)
}

func (e *GolangBGE3M3Embedder) Upsert(id, text string, vec []float32, meta map[string]interface{}) {
    e.VecStore.Upsert(id, text, e.VecStore.F32ToF64(vec), meta)
}

func (e *GolangBGE3M3Embedder) SaveJSON(path string) error {
    if err := e.VecStore.SaveJSON(MEMORY_PATH); err != nil {
        return err
    }
    return nil
}

func (e GolangBGE3M3Embedder) LoadJSON() (*model.VecStore, error) {
    return e.VecStore.LoadJSON(MEMORY_PATH)
}

func (e *GolangBGE3M3Embedder) SearchVector(queryText string, queryVec []float32, dims, topK int) ([]co.OrderedPair[string, model.EmbeddingRecord], error) {
    return e.VecStore.SearchVector(queryText, e.VecStore.F32ToF64(queryVec), dims, topK)
}

func (e *GolangBGE3M3Embedder) SearchVectorFiltered(queryText string, queryVec []float32, dims, topK int, filter string) ([]co.OrderedPair[string, model.EmbeddingRecord], error) {
    q := make([]float64, 1024)
    q[0] = 0.7
    loaded, err := e.LoadJSON()
    if err != nil {
        return nil, err
    }
    top, err := loaded.SearchVector(queryText, q, 1024, 5)
    if e.Verbose {
        fmt.Printf("Top %d results for query: %s\n", topK, queryText)
        for i, p := range top {
            fmt.Printf("%d) id=%s  text=%q  dim=%d\n", i+1, p.Key, p.Value.Text, len(p.Value.Vector))
            fmt.Printf("Score: %f\n", p.Value.Vector[0])
            fmt.Printf("Meta: %v\n", p.Value.Meta)
            fmt.Printf("CreatedAt: %v\n", p.Value.CreatedAt)
            fmt.Printf("Vector: %v\n", p.Value.Vector)
        }
    }
    if err != nil {
        return nil, err
    }
    return top, nil
}

