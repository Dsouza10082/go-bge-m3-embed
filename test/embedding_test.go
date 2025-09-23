// test/embedding_test.go
package test

import (
	"testing"
	"math"
	bge "github.com/Dsouza10082/go-bge-m3-embed"
)

func TestNewEmbedder(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
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
	
	testCases := []struct {
		name string
		text string
		expectedDim int
	}{
		{"Simple text", "Hello world", 1024},
		{"Long text", "This is a much longer text that should still produce a 1024-dimensional embedding vector", 1024},
		{"Empty text", "", 1024},
		{"Special characters", "Hello, 世界! 🌍", 1024},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			vector := embedder.Embed(tc.text)
			if len(vector) != tc.expectedDim {
				t.Errorf("Expected dimension %d, got %d", tc.expectedDim, len(vector))
			}
			
			// Check if vector is properly normalized (L2 norm ≈ 1.0)
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
	
	texts := []string{
		"First document",
		"Second document", 
		"Third document",
	}
	
	vectors := embedder.EmbedBatch(texts)
	
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
	
	// Test data
	text := "Test document for storage"
	vector := embedder.Embed(text)
	meta := map[string]interface{}{
		"category": "test",
		"importance": 0.8,
	}
	
	// Test upsert
	embedder.Upsert("test_001", text, vector, meta)
	
	// Test search
	results, err := embedder.SearchVector(text, vector, 1024, 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	
	if len(results) == 0 {
		t.Fatal("No results found")
	}
	
	// Should find exact match with high similarity
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
	
	// Add test data
	testData := map[string]string{
		"doc_001": "First test document",
		"doc_002": "Second test document",
		"doc_003": "Third test document",
	}
	
	for id, text := range testData {
		vector := embedder.Embed(text)
		meta := map[string]interface{}{"test": true}
		embedder.Upsert(id, text, vector, meta)
	}
	
	// Test save
	err := embedder.SaveJSON("./test_vecstore.json")
	if err != nil {
		t.Fatalf("Failed to save: %v", err)
	}
	
	// Test load
	newEmbedder := bge.NewGolangBGE3M3Embedder()
	_, err = newEmbedder.LoadJSON()
	if err != nil {
		t.Fatalf("Failed to load: %v", err)
	}
	
	// Clean up
	// os.Remove("./test_vecstore.json")
}

func TestCosineSimiliarity(t *testing.T) {
	embedder := bge.NewGolangBGE3M3Embedder()
	
	// Similar texts should have high similarity
	text1 := "The cat sits on the mat"
	text2 := "A cat is sitting on the mat"
	
	vec1 := embedder.Embed(text1)
	vec2 := embedder.Embed(text2)
	
	similarity := embedder.EmbeddingModel.Cosine(vec1, vec2)
	
	if similarity < 0.7 {
		t.Errorf("Expected high similarity for similar texts, got %f", similarity)
	}
	
	// Dissimilar texts should have lower similarity
	text3 := "Quantum mechanics and relativity theory"
	vec3 := embedder.Embed(text3)
	
	similarity2 := embedder.EmbeddingModel.Cosine(vec1, vec3)
	
	if similarity2 > similarity {
		t.Errorf("Expected lower similarity for dissimilar texts")
	}
}