// benchmark/embedding_bench_test.go
package benchmark

import (
	"testing"
	"fmt"
	bge "github.com/Dsouza10082/go-bge-m3-embed"
)

var embedder *bge.GolangBGE3M3Embedder

func init() {
	embedder = bge.NewGolangBGE3M3Embedder()
}

func BenchmarkSingleEmbedding(b *testing.B) {
	text := "This is a test document for benchmarking single embedding performance"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = embedder.Embed(text)
	}
}

func BenchmarkBatchEmbedding10(b *testing.B) {
	texts := make([]string, 10)
	for i := 0; i < 10; i++ {
		texts[i] = fmt.Sprintf("Document %d for batch embedding benchmark", i)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = embedder.EmbedBatch(texts)
	}
}

func BenchmarkBatchEmbedding100(b *testing.B) {
	texts := make([]string, 100)
	for i := 0; i < 100; i++ {
		texts[i] = fmt.Sprintf("Document %d for batch embedding benchmark", i)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = embedder.EmbedBatch(texts)
	}
}

func BenchmarkVectorSearch1K(b *testing.B) {
	// Setup: Create 1000 vectors
	for i := 0; i < 1000; i++ {
		text := fmt.Sprintf("Document %d content for search benchmarking", i)
		vector := embedder.Embed(text)
		meta := map[string]interface{}{"index": i}
		embedder.Upsert(fmt.Sprintf("doc_%d", i), text, vector, meta)
	}
	
	queryText := "search query for benchmarking"
	queryVec := embedder.Embed(queryText)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = embedder.SearchVector(queryText, queryVec, 1024, 10)
	}
}

func BenchmarkVectorSearch10K(b *testing.B) {
	// Setup: Create 10000 vectors
	for i := 0; i < 10000; i++ {
		text := fmt.Sprintf("Document %d content for large scale search benchmarking", i)
		vector := embedder.Embed(text)
		meta := map[string]interface{}{"index": i}
		embedder.Upsert(fmt.Sprintf("doc_%d", i), text, vector, meta)
	}
	
	queryText := "search query for large scale benchmarking"
	queryVec := embedder.Embed(queryText)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = embedder.SearchVector(queryText, queryVec, 1024, 10)
	}
}

func BenchmarkConcurrentEmbedding(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			text := "Concurrent embedding benchmark text"
			_ = embedder.Embed(text)
		}
	})
}

func BenchmarkConcurrentSearch(b *testing.B) {
	// Setup vectors for search
	for i := 0; i < 1000; i++ {
		text := fmt.Sprintf("Document %d for concurrent search", i)
		vector := embedder.Embed(text)
		embedder.Upsert(fmt.Sprintf("doc_%d", i), text, vector, nil)
	}
	
	queryText := "concurrent search query"
	queryVec := embedder.Embed(queryText)
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = embedder.SearchVector(queryText, queryVec, 1024, 5)
		}
	})
}