package model

import (
	"encoding/json"
	"errors"
	"os"
	"time"

	co "github.com/Dsouza10082/ConcurrentOrderedMap"
)

type EmbeddingRecord struct {
	ID        string                 `json:"id"`
	Text      string                 `json:"text"`
	Vector    []float64              `json:"vector"` // 1024-D (BGE-M3)
	CreatedAt time.Time              `json:"created_at,omitempty"`
	Meta      map[string]interface{} `json:"meta,omitempty"`
}

type VecStore struct {
	M *co.ConcurrentOrderedMap[string, EmbeddingRecord]
}

func NewVecStore() *VecStore {
	return &VecStore{
		M: co.NewConcurrentOrderedMap[string, EmbeddingRecord](),
	}
}

func (s *VecStore) Upsert(id, text string, vec []float64, meta map[string]interface{}) {
	rec := EmbeddingRecord{
		ID:        id,
		Text:      text,
		Vector:    vec,
		CreatedAt: time.Now(),
		Meta:      meta,
	}
	s.M.Set(id, rec)
}

func (s *VecStore) SaveJSON(path string) error {
	pairs := s.M.GetOrderedV2()
	out := make([]EmbeddingRecord, 0, len(pairs))
	for _, p := range pairs {
		out = append(out, p.Value)
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func (s *VecStore) LoadJSON(path string) (*VecStore, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var items []EmbeddingRecord
	if err := json.NewDecoder(f).Decode(&items); err != nil {
		return nil, err
	}
	v := NewVecStore()
	for _, it := range items {
		v.M.Set(it.ID, it)
	}
	return s, nil
}

func (s *VecStore) RecordExtractor(v EmbeddingRecord) (string, []float64, bool) {
	if len(v.Vector) == 0 {
		return v.Text, nil, false
	}
	return v.Text, v.Vector, true
}

func (s *VecStore) SearchVector(queryText string, queryVec []float64, dims, topK int) ([]co.OrderedPair[string, EmbeddingRecord], error) {
	if len(queryVec) != dims {
		return nil, errors.New("queryVec wrong dimension")
	}
	w := co.DefaultVectorBlendWeights() 
	pairs, err := s.M.OrderedByVectorCombinedSimilarity(
		queryText,           
		true,                
		queryVec,          
		dims,               
		s.RecordExtractor,     
		&w,                  
	)
	if err != nil {
		return nil, err
	}
	if topK > 0 && topK < len(pairs) {
		pairs = pairs[:topK]
	}
	return pairs, nil
}

func (s *VecStore) F32ToF64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, x := range in {
		out[i] = float64(x)
	}
	return out
}