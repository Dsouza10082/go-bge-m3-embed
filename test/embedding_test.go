package test

import (
	"testing"
	"math"
	"os"
	"encoding/json"
	bge "github.com/Dsouza10082/go-bge-m3-embed"
)

func TestNewEmbedder(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	if embedder == nil {
		t.Fatal("Failed to create new embedder")
	}
	if embedder.EmbeddingModel == nil {
		t.Fatal("EmbeddingModel not initialized")
	}
	if embedder.VecStore == nil {
		t.Fatal("VecStore not initialized")
	}
}

func TestEmbedding(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	testCases := []struct {
		name string
		text string
		expectedDim int
	}{
		{"Simple text", "Hello world", 1024},
		{"Long text", "This is a much longer text that should still produce a 1024-dimensional embedding vector", 1024},
		{"Special characters", "Hello, 世界! 🌍", 1024},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			vector, err := embedder.Embed(tc.text)
			if err != nil {
				t.Errorf("Error embedding text: %v", err)
			}
			if len(vector) != tc.expectedDim {
				t.Errorf("Expected dimension %d, got %d", tc.expectedDim, len(vector))
			}
			
			var norm float64
			for _, v := range vector {
				norm += float64(v) * float64(v)
			}
			norm = math.Sqrt(norm)
			if math.Abs(norm - 1.0) > 0.01 {
				t.Errorf("Vector not properly normalized. L2 norm: %f", norm)
			}
		})
	}
}

func TestEmbeddingValidation(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	
	_, err := embedder.Embed("")
	if err == nil {
		t.Error("Expected error for empty text, got nil")
	}
}

func TestBatchEmbedding(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	texts := []string{
		"First document",
		"Second document", 
		"Third document",
	}
	
	vectors, err := embedder.EmbedBatch(texts)
	if err != nil {
		t.Fatalf("Error embedding batch: %v", err)
	}
	
	if len(vectors) != len(texts) {
		t.Errorf("Expected %d vectors, got %d", len(texts), len(vectors))
	}
	
	for i, vec := range vectors {
		if len(vec) != 1024 {
			t.Errorf("Vector %d has wrong dimension: %d", i, len(vec))
		}
	}
}

func TestBatchEmbeddingValidation(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	
	_, err := embedder.EmbedBatch([]string{})
	if err == nil {
		t.Error("Expected error for empty texts slice, got nil")
	}
}

func TestVectorStorage(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()

	embedder.SetMemoryPath("./test_vecstore")
	text := "Test document for storage"
	vector, err := embedder.Embed(text)
	if err != nil {
		t.Fatalf("Error embedding text: %v", err)
	}
	meta := map[string]interface{}{
		"category": "test",
		"importance": 0.8,
	}
	
	embedder.Upsert("test_001", text, vector, meta)
	
	results, err := embedder.SearchVector(text, vector, 1024, 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	
	if len(results) == 0 {
		t.Fatal("No results found")
	}
	
	topResult := results[0]
	if topResult.Key != "test_001" {
		t.Errorf("Expected key 'test_001', got '%s'", topResult.Key)
	}
	
	if topResult.Value.Text != text {
		t.Errorf("Expected text '%s', got '%s'", text, topResult.Value.Text)
	}
}

func TestUpsertWithValidation(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	
	text := "Test document with validation"
	vector, err := embedder.Embed(text)
	if err != nil {
		t.Fatalf("Error embedding text: %v", err)
	}
	
	meta := map[string]interface{}{"validated": true}
	
	err = embedder.UpsertWithValidation("test_validated_001", text, vector, meta)
	if err != nil {
		t.Errorf("UpsertWithValidation failed: %v", err)
	}
	
	err = embedder.UpsertWithValidation("", text, vector, meta)
	if err == nil {
		t.Error("Expected error for empty ID, got nil")
	}
	
	err = embedder.UpsertWithValidation("test_002", "", vector, meta)
	if err == nil {
		t.Error("Expected error for empty text, got nil")
	}
	
	err = embedder.UpsertWithValidation("test_003", text, []float32{}, meta)
	if err == nil {
		t.Error("Expected error for empty vector, got nil")
	}
	
	err = embedder.UpsertWithValidation("test_004", text, vector, nil)
	if err != nil {
		t.Errorf("Should handle nil metadata, got error: %v", err)
	}
}

func TestVerifyUpsert(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	
	text := "Test document for verification"
	vector, err := embedder.Embed(text)
	if err != nil {
		t.Fatalf("Error embedding text: %v", err)
	}
	
	testID := "verify_test_001"
	err = embedder.UpsertWithValidation(testID, text, vector, nil)
	if err != nil {
		t.Fatalf("UpsertWithValidation failed: %v", err)
	}
	
	exists, err := embedder.VerifyUpsert(testID)
	if err != nil {
		t.Errorf("VerifyUpsert failed: %v", err)
	}
	if !exists {
		t.Error("Expected record to exist after upsert")
	}
	
	_, err = embedder.VerifyUpsert("")
	if err == nil {
		t.Error("Expected error for empty ID, got nil")
	}
}

func TestUpsertBGEM3Row(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	
	row := &bge.BGEM3Row{
		SerialId:       "row_001",
		SerialMasterId: "master_001",
		Text:           "Test document for BGEM3Row",
		Score:          0.95,
		Metadata: map[string]interface{}{
			"category": "test",
			"priority": "high",
		},
	}
	
	result, err := embedder.UpsertBGEM3Row(row)
	if err != nil {
		t.Fatalf("UpsertBGEM3Row failed: %v", err)
	}
	
	if result.Status != "success" {
		t.Errorf("Expected status 'success', got '%s'", result.Status)
	}
	
	if result.RowError != "" {
		t.Errorf("Expected empty RowError, got '%s'", result.RowError)
	}
	
	if result.Created == "" {
		t.Error("Created timestamp should be set")
	}
	
	if result.Metadata["serial_master_id"] != "master_001" {
		t.Error("Metadata should contain serial_master_id")
	}
}

func TestUpsertBGEM3RowValidation(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	
	_, err := embedder.UpsertBGEM3Row(nil)
	if err == nil {
		t.Error("Expected error for nil row, got nil")
	}
	
	emptyIDRow := &bge.BGEM3Row{
		Text: "Test text",
	}
	result, err := embedder.UpsertBGEM3Row(emptyIDRow)
	if err == nil {
		t.Error("Expected error for empty SerialId, got nil")
	}
	if result.Status != "failed" {
		t.Error("Expected status 'failed' for validation error")
	}
	if result.RowError == "" {
		t.Error("Expected RowError to be set")
	}
	
	emptyTextRow := &bge.BGEM3Row{
		SerialId: "row_002",
	}
	result, err = embedder.UpsertBGEM3Row(emptyTextRow)
	if err == nil {
		t.Error("Expected error for empty Text, got nil")
	}
	if result.Status != "failed" {
		t.Error("Expected status 'failed' for validation error")
	}
}

func TestUpsertBGEM3RowBatch(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	
	rows := []*bge.BGEM3Row{
		{
			SerialId:       "batch_001",
			SerialMasterId: "master_batch",
			Text:           "First batch document",
			Score:          0.9,
		},
		{
			SerialId:       "batch_002",
			SerialMasterId: "master_batch",
			Text:           "Second batch document",
			Score:          0.85,
		},
		{
			SerialId: "batch_003",
			Text:     "Third batch document",
		},
	}
	
	results, errs := embedder.UpsertBGEM3RowBatch(rows)
	
	if len(results) != len(rows) {
		t.Errorf("Expected %d results, got %d", len(rows), len(results))
	}
	
	successCount := 0
	for i, result := range results {
		if result.Status == "success" {
			successCount++
			if errs[i] != nil {
				t.Errorf("Row %d has success status but non-nil error", i)
			}
		}
	}
	
	if successCount != len(rows) {
		t.Logf("Processed %d successful rows out of %d", successCount, len(rows))
	}
}

func TestUpsertBGEM3RowBatchValidation(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	
	_, errs := embedder.UpsertBGEM3RowBatch([]*bge.BGEM3Row{})
	if len(errs) == 0 || errs[0] == nil {
		t.Error("Expected error for empty rows slice")
	}
}

func TestExportBGEM3RowsToJSON(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore_export")
	
	rows := []*bge.BGEM3Row{
		{
			SerialId:       "export_001",
			SerialMasterId: "master_export",
			Text:           "First export document",
			Score:          0.92,
			Metadata: map[string]interface{}{
				"type": "export_test",
			},
		},
		{
			SerialId:       "export_002",
			SerialMasterId: "master_export",
			Text:           "Second export document",
			Score:          0.88,
		},
	}
	
	_, _ = embedder.UpsertBGEM3RowBatch(rows)
	
	err := embedder.SaveJSON()
	if err != nil {
		t.Fatalf("Failed to save: %v", err)
	}
	
	jsonData, err := embedder.ExportBGEM3RowsToJSON()
	if err != nil {
		t.Fatalf("ExportBGEM3RowsToJSON failed: %v", err)
	}
	
	if len(jsonData) == 0 {
		t.Error("Expected non-empty JSON data")
	}
	
	var exportedRows []*bge.BGEM3Row
	err = json.Unmarshal(jsonData, &exportedRows)
	if err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}
	
	if len(exportedRows) < len(rows) {
		t.Errorf("Expected at least %d exported rows, got %d", len(rows), len(exportedRows))
	}
	
	for _, row := range exportedRows {
		if row.Status != "exported" {
			t.Errorf("Expected status 'exported', got '%s'", row.Status)
		}
	}
}

func TestPersistence(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	testData := map[string]string{
		"doc_001": "First test document",
		"doc_002": "Second test document",
		"doc_003": "Third test document",
	}
	
	for id, text := range testData {
		vector, err := embedder.Embed(text)
		if err != nil {
			t.Fatalf("Error embedding text: %v", err)
		}
		meta := map[string]interface{}{"test": true}
		embedder.Upsert(id, text, vector, meta)
	}
	
	err := embedder.SaveJSON()
	if err != nil {
		t.Fatalf("Failed to save: %v", err)
	}
	
	newEmbedder := bge.NewGolangBGE3M3Embedder()
	newEmbedder.SetMemoryPath("./test_vecstore")
	_, err = newEmbedder.LoadJSON()
	if err != nil {
		t.Fatalf("Failed to load: %v", err)
	}
}

func TestSearchVectorValidation(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	
	_, err := embedder.SearchVector("", []float32{}, 1024, 5)
	if err == nil {
		t.Error("Expected error when both queryText and queryVec are empty")
	}
	
	vector, _ := embedder.Embed("test")
	_, err = embedder.SearchVector("test", vector, 0, 5)
	if err == nil {
		t.Error("Expected error for dims <= 0")
	}
	
	_, err = embedder.SearchVector("test", vector, 1024, 0)
	if err == nil {
		t.Error("Expected error for topK <= 0")
	}
}

func TestCosineSimiliarity(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore")
	text1 := "The cat sits on the mat"
	text2 := "A cat is sitting on the mat"
	
	vec1, err := embedder.Embed(text1)
	if err != nil {
		t.Fatalf("Error embedding text: %v", err)
	}
	vec2, err := embedder.Embed(text2)
	if err != nil {
		t.Fatalf("Error embedding text: %v", err)
	}
	
	similarity := embedder.EmbeddingModel.Cosine(vec1, vec2)
	
	if similarity < 0.7 {
		t.Errorf("Expected high similarity for similar texts, got %f", similarity)
	}
	
	text3 := "Quantum mechanics and relativity theory"
	vec3, err := embedder.Embed(text3)
	if err != nil {
		t.Fatalf("Error embedding text: %v", err)
	}
	
	similarity2 := embedder.EmbeddingModel.Cosine(vec1, vec3)
	
	if similarity2 > similarity {
		t.Errorf("Expected lower similarity for dissimilar texts")
	}
}

func TestBGEM3RowMetadataPreservation(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	embedder.SetMemoryPath("./test_vecstore_meta")
	
	customMeta := map[string]interface{}{
		"author":   "Test Author",
		"version":  1,
		"tags":     []string{"test", "metadata"},
		"priority": 10,
	}
	
	row := &bge.BGEM3Row{
		SerialId:       "meta_test_001",
		SerialMasterId: "meta_master",
		Text:           "Document with custom metadata",
		Score:          0.96,
		Metadata:       customMeta,
	}
	
	result, err := embedder.UpsertBGEM3Row(row)
	if err != nil {
		t.Fatalf("UpsertBGEM3Row failed: %v", err)
	}
	
	if result.Metadata["author"] != "Test Author" {
		t.Error("Custom metadata 'author' not preserved")
	}
	
	if result.Metadata["serial_master_id"] != "meta_master" {
		t.Error("serial_master_id not added to metadata")
	}
	
	if result.Metadata["score"] != 0.96 {
		t.Error("score not added to metadata")
	}
	
	if result.Metadata["created"] == nil {
		t.Error("created timestamp not added to metadata")
	}
}

func TestCleanup(t *testing.T) {
	testPaths := []string{
		"./test_vecstore",
		"./test_vecstore_export",
		"./test_vecstore_meta",
	}
	
	for _, path := range testPaths {
		err := os.RemoveAll(path)
		if err != nil && !os.IsNotExist(err) {
			t.Logf("Warning: Failed to cleanup %s: %v", path, err)
		}
	}
}