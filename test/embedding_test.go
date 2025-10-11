package test

import (
	"testing"
	"math"
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