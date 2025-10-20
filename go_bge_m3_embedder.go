package golangbgem3embedder

// Golang-BGE-M3-embedder
//
// Author: Dsouza
// License: MIT

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"time"

	co "github.com/Dsouza10082/ConcurrentOrderedMap"
	"github.com/Dsouza10082/go-bge-m3-embed/model"
)


// BGEM3Row represents a structured row for embedding content with metadata and status tracking
type BGEM3Row struct {
	Metadata       map[string]interface{} `json:"metadata"`
	Text           string                 `json:"text"`
	Score          float64                `json:"score"`
	SerialId       string                 `json:"serial_id"`
	SerialMasterId string                 `json:"serial_master_id"`
	Created        string                 `json:"created"`
	Status         string                 `json:"status"`
	RowError       string                 `json:"row_error"`
}

type GolangBGE3M3Embedder struct {
	EmbeddingModel *model.EmbeddingModel
	VecStore       *model.VecStore
	Verbose        bool
	memoryPath     string
}

// NewGolangBGE3M3Embedder creates a new embedder instance with default configuration
//
// Example:
//
//	embedder := NewGolangBGE3M3Embedder()
//	embedder.SetMemoryPath("./my_memory")
func NewGolangBGE3M3Embedder() *GolangBGE3M3Embedder {
	return &GolangBGE3M3Embedder{
		EmbeddingModel: model.NewEmbeddingModel(),
		VecStore:       model.NewVecStore(),
		Verbose:        false,
	}
}

// Embed generates embeddings for a single text string
//
// Example:
//
//	embedder := NewGolangBGE3M3Embedder()
//	vec, err := embedder.Embed("Hello world")
//	if err != nil {
//	    log.Fatal(err)
//	}
func (e *GolangBGE3M3Embedder) Embed(text string) ([]float32, error) {
	if text == "" {
		return nil, errors.New("text cannot be empty")
	}

	tk, err := e.EmbeddingModel.NewTokenizer()
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizer: %w", err)
	}

	vec, err := e.EmbeddingModel.Embed(tk, text)
	if err != nil {
		return nil, fmt.Errorf("failed to embed text: %w", err)
	}

	return vec, nil
}

// SetMemoryPath configures the path where vector store data will be saved
// Important: for while is mandatory use the name "vec_store.json" as file store name
// Example:
//
//	embedder := NewGolangBGE3M3Embedder().SetMemoryPath("./custom_memory/vec_store.json")
func (e *GolangBGE3M3Embedder) SetMemoryPath(path string) *GolangBGE3M3Embedder {
	e.memoryPath = path
	return e
}

// EmbedBatch generates embeddings for multiple text strings in a single batch
//
// Example:
//
//	texts := []string{"Hello world", "Goodbye world"}
//	vecs, err := embedder.EmbedBatch(texts)
//	if err != nil {
//	    log.Fatal(err)
//	}
func (e *GolangBGE3M3Embedder) EmbedBatch(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, errors.New("texts slice cannot be empty")
	}

	tk, err := e.EmbeddingModel.NewTokenizer()
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizer: %w", err)
	}

	vecs, err := e.EmbeddingModel.EmbedBatch(tk, texts)
	if err != nil {
		return nil, fmt.Errorf("failed to embed batch: %w", err)
	}

	return vecs, nil
}

// EmbedBGE3MText generates embeddings using the BGE-M3 model for a single text
//
// Example:
//
//	vec := embedder.EmbedBGE3MText("Sample text")
func (e *GolangBGE3M3Embedder) EmbedBGE3MText(text string) []float32 {
	return e.EmbeddingModel.EmbedBGE3MText(text)
}

// Upsert stores or updates a vector embedding with metadata (deprecated, use UpsertWithValidation)
//
// Example:
//
//	vec, _ := embedder.Embed("Hello world")
//	meta := map[string]interface{}{"source": "example"}
//	embedder.Upsert("id1", "Hello world", vec, meta)
func (e *GolangBGE3M3Embedder) Upsert(id, text string, vec []float32, meta map[string]interface{}) {
	e.VecStore.Upsert(id, text, e.VecStore.F32ToF64(vec), meta)
}

// UpsertWithValidation stores or updates a vector embedding with validation and error handling
//
// Example:
//
//	vec, _ := embedder.Embed("Hello world")
//	meta := map[string]interface{}{"source": "example"}
//	err := embedder.UpsertWithValidation("id1", "Hello world", vec, meta)
//	if err != nil {
//	    log.Printf("Failed to upsert: %v", err)
//	}
func (e *GolangBGE3M3Embedder) UpsertWithValidation(id, text string, vec []float32, meta map[string]interface{}) error {
	if id == "" {
		return errors.New("id cannot be empty")
	}
	if text == "" {
		return errors.New("text cannot be empty")
	}
	if len(vec) == 0 {
		return errors.New("vector cannot be empty")
	}
	if meta == nil {
		meta = make(map[string]interface{})
	}

	vec64 := e.VecStore.F32ToF64(vec)
	e.VecStore.Upsert(id, text, vec64, meta)

	if e.VecStore == nil {
		return errors.New("vector store is nil after upsert")
	}

	return nil
}

// UpsertBGEM3Row stores a BGEM3Row with full validation and status tracking
//
// Example:
//
//	row := &BGEM3Row{
//	    SerialId: "row1",
//	    Text: "Sample text",
//	    Metadata: map[string]interface{}{"category": "test"},
//	}
//	result, err := embedder.UpsertBGEM3Row(row)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Status: %s\n", result.Status)
func (e *GolangBGE3M3Embedder) UpsertBGEM3Row(row *BGEM3Row) (*BGEM3Row, error) {
	if row == nil {
		return nil, errors.New("row cannot be nil")
	}

	if row.SerialId == "" {
		row.Status = "failed"
		row.RowError = "serial_id is required"
		return row, errors.New("serial_id is required")
	}

	if row.Text == "" {
		row.Status = "failed"
		row.RowError = "text is required"
		return row, errors.New("text is required")
	}

	if row.Created == "" {
		row.Created = time.Now().UTC().Format(time.RFC3339)
	}

	if row.Metadata == nil {
		row.Metadata = make(map[string]interface{})
	}

	row.Metadata["serial_master_id"] = row.SerialMasterId
	row.Metadata["created"] = row.Created
	row.Metadata["score"] = row.Score

	vec, err := e.Embed(row.Text)
	if err != nil {
		row.Status = "failed"
		row.RowError = fmt.Sprintf("embedding failed: %v", err)
		return row, fmt.Errorf("failed to generate embedding: %w", err)
	}

	err = e.UpsertWithValidation(row.SerialId, row.Text, vec, row.Metadata)
	if err != nil {
		row.Status = "failed"
		row.RowError = fmt.Sprintf("upsert failed: %v", err)
		return row, fmt.Errorf("failed to upsert: %w", err)
	}

	row.Status = "success"
	row.RowError = ""

	return row, nil
}

// UpsertBGEM3RowBatch processes multiple BGEM3Row entries in batch with individual error tracking
//
// Example:
//
//	rows := []*BGEM3Row{
//	    {SerialId: "row1", Text: "First text"},
//	    {SerialId: "row2", Text: "Second text"},
//	}
//	results, errors := embedder.UpsertBGEM3RowBatch(rows)
//	for i, result := range results {
//	    fmt.Printf("Row %d status: %s\n", i, result.Status)
//	}
func (e *GolangBGE3M3Embedder) UpsertBGEM3RowBatch(rows []*BGEM3Row) ([]*BGEM3Row, []error) {
	if len(rows) == 0 {
		return nil, []error{errors.New("rows slice cannot be empty")}
	}

	results := make([]*BGEM3Row, len(rows))
	errs := make([]error, len(rows))

	for i, row := range rows {
		result, err := e.UpsertBGEM3Row(row)
		results[i] = result
		errs[i] = err
	}

	return results, errs
}

// VerifyUpsert checks if a record exists in the vector store by ID
//
// Example:
//
//	exists, err := embedder.VerifyUpsert("id1")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	if exists {
//	    fmt.Println("Record exists")
//	}
func (e *GolangBGE3M3Embedder) VerifyUpsert(id string) (bool, error) {
	if id == "" {
		return false, errors.New("id cannot be empty")
	}

	results, err := e.VecStore.SearchVector(id, nil, 1024, 1)
	if err != nil {
		return false, fmt.Errorf("failed to verify upsert: %w", err)
	}

	for _, result := range results {
		if result.Key == id {
			return true, nil
		}
	}

	return false, nil
}

// SaveJSON persists the vector store to disk in JSON format
//
// Example:
//
//	err := embedder.SaveJSON()
//	if err != nil {
//	    log.Fatal(err)
//	}
func (e *GolangBGE3M3Embedder) SaveJSON() error {
	if err := e.VecStore.SaveJSON(filepath.Join(e.memoryPath, "vec_store.json")); err != nil {
		return fmt.Errorf("failed to save JSON: %w", err)
	}
	return nil
}

// LoadJSON loads the vector store from disk
//
// Example:
//
//	vecStore, err := embedder.LoadJSON()
//	if err != nil {
//	    log.Fatal(err)
//	}
func (e GolangBGE3M3Embedder) LoadJSON() (*model.VecStore, error) {
	vecStore, err := e.VecStore.LoadJSON(filepath.Join(e.memoryPath, "vec_store.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to load JSON: %w", err)
	}
	return vecStore, nil
}

// SearchVector performs a vector similarity search
//
// Example:
//
//	queryVec, _ := embedder.Embed("search query")
//	results, err := embedder.SearchVector("search query", queryVec, 1024, 5)
//	if err != nil {
//	    log.Fatal(err)
//	}
func (e *GolangBGE3M3Embedder) SearchVector(queryText string, queryVec []float32, dims, topK int) ([]co.OrderedPair[string, model.EmbeddingRecord], error) {
	if queryText == "" && len(queryVec) == 0 {
		return nil, errors.New("either queryText or queryVec must be provided")
	}
	if dims <= 0 {
		return nil, errors.New("dims must be greater than 0")
	}
	if topK <= 0 {
		return nil, errors.New("topK must be greater than 0")
	}

	results, err := e.VecStore.SearchVector(queryText, e.VecStore.F32ToF64(queryVec), dims, topK)
	if err != nil {
		return nil, fmt.Errorf("vector search failed: %w", err)
	}

	return results, nil
}

// SearchVectorFiltered performs a filtered vector similarity search
//
// Example:
//
//	queryVec, _ := embedder.Embed("search query")
//	results, err := embedder.SearchVectorFiltered("search query", queryVec, 1024, 5, "category=test")
//	if err != nil {
//	    log.Fatal(err)
//	}
func (e *GolangBGE3M3Embedder) SearchVectorFiltered(queryText string, queryVec []float32, dims, topK int, filter string) ([]co.OrderedPair[string, model.EmbeddingRecord], error) {
	q := make([]float64, 1024)
	q[0] = 0.7
	loaded, err := e.LoadJSON()
	if err != nil {
		return nil, fmt.Errorf("failed to load vector store: %w", err)
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
		return nil, fmt.Errorf("filtered search failed: %w", err)
	}
	return top, nil
}

// ExportBGEM3RowsToJSON exports all vector store entries as BGEM3Row format to JSON
//
// Example:
//
//	jsonData, err := embedder.ExportBGEM3RowsToJSON()
//	if err != nil {
//	    log.Fatal(err)
//	}
//	os.WriteFile("export.json", jsonData, 0644)
func (e *GolangBGE3M3Embedder) ExportBGEM3RowsToJSON() ([]byte, error) {
	vecStore, err := e.LoadJSON()
	if err != nil {
		return nil, fmt.Errorf("failed to load vector store: %w", err)
	}

	results, err := vecStore.SearchVector("", nil, 1024, 10000)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve all vectors: %w", err)
	}

	rows := make([]*BGEM3Row, 0, len(results))
	for _, result := range results {
		row := &BGEM3Row{
			SerialId: result.Key,
			Text:     result.Value.Text,
			Metadata: result.Value.Meta,
			Created:  result.Value.CreatedAt.Format(time.RFC3339),
			Status:   "exported",
			RowError: "",
		}

		if masterId, ok := result.Value.Meta["serial_master_id"].(string); ok {
			row.SerialMasterId = masterId
		}
		if score, ok := result.Value.Meta["score"].(float64); ok {
			row.Score = score
		}

		rows = append(rows, row)
	}

	jsonData, err := json.MarshalIndent(rows, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal to JSON: %w", err)
	}

	return jsonData, nil
}

// SetOnnxPath configures the path where the ONNX model is located
//
// Example:
//
//	embedder := NewGolangBGE3M3Embedder().SetOnnxPath("./custom_onnx/model.onnx")
func (e *GolangBGE3M3Embedder) SetOnnxPath(path string) *GolangBGE3M3Embedder {
	absPath, _ := filepath.Abs(path)
	e.EmbeddingModel.SetOnnxPath(absPath)
	if e.Verbose {
		fmt.Printf("[%s]-golangbgem3embedder SetOnnxPath: %s\n", time.Now().Format(time.RFC3339), e.EmbeddingModel.OnnxPath)
	}
	return e
}

// SetTokPath configures the path where the tokenizer is located
//
// Example:
//
//	embedder := NewGolangBGE3M3Embedder().SetTokPath("./custom_tok/tokenizer.json")
func (e *GolangBGE3M3Embedder) SetTokPath(path string) *GolangBGE3M3Embedder {
	absPath, _ := filepath.Abs(path)
	e.EmbeddingModel.SetTokPath(absPath)
	if e.Verbose {
		fmt.Printf("[%s]-golangbgem3embedder SetTokPath: %s\n", time.Now().Format(time.RFC3339), e.EmbeddingModel.TokPath)
	}
	return e
}


// SetRuntimePath configures the path where the runtime is located
//
// Example:
//
//	embedder := NewGolangBGE3M3Embedder().SetRuntimePath("./custom_runtime/libonnxruntime.dylib")
func (e *GolangBGE3M3Embedder) SetRuntimePath(path string) *GolangBGE3M3Embedder {
	absPath, _ := filepath.Abs(path)
	e.EmbeddingModel.SetRuntimePath(absPath)
	if e.Verbose {
		fmt.Printf("[%s]-golangbgem3embedder SetRuntimePath: %s\n", time.Now().Format(time.RFC3339), e.EmbeddingModel.RuntimePath)
	}
	return e
}
