package golangbgem3embedder

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	co "github.com/Dsouza10082/ConcurrentOrderedMap"
)

type Node struct {
	ID       string    `json:"id"`
	Label    string    `json:"label"`
	Type     string    `json:"type"`
	Vector   []float32 `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Created  time.Time `json:"created"`
	Updated  time.Time `json:"updated"`
}

type Link struct {
	Source   string                 `json:"source"`
	Target   string                 `json:"target"`
	Label    string                 `json:"label"`
	Weight   float32                `json:"weight,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

type Graph struct {
	Nodes []Node `json:"nodes"`
	Links []Link `json:"links"`
}

type SearchResult struct {
	Node       Node    `json:"node"`
	Similarity float32 `json:"similarity"`
}

type KnowledgeGraphServer struct {
	nodes     *co.ConcurrentOrderedMap[string, Node]
	links     *co.ConcurrentOrderedMap[string, Link]
	embedder  *GolangBGE3M3Embedder
	mu        sync.RWMutex
	vectorDim int
}

func NewKnowledgeGraphServer() (*KnowledgeGraphServer, error) {
	embedder := NewGolangBGE3M3Embedder()
	if embedder == nil {
		log.Printf("Warning: Failed to initialize embedder. Using mock embedder.")
		embedder = nil
	}
	return &KnowledgeGraphServer{
		nodes:     co.NewConcurrentOrderedMap[string, Node](),
		links:     co.NewConcurrentOrderedMap[string, Link](),
		embedder:  embedder,
		vectorDim: 768,
	}, nil
}

func (kgs *KnowledgeGraphServer) generateEmbedding(text string) ([]float32, error) {
	if kgs.embedder != nil {
		return kgs.embedder.Embed(text)
	}
	vector := make([]float32, kgs.vectorDim)
	for i := range vector {
		vector[i] = float32(math.Sin(float64(i)) * math.Cos(float64(len(text))))
	}
	return vector, nil
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func (kgs *KnowledgeGraphServer) AddNode(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID           string                 `json:"id"`
		Label        string                 `json:"label"`
		Type         string                 `json:"type"`
		EmbeddingText string                `json:"embedding_text"`
		Metadata     map[string]interface{} `json:"metadata"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	vector, err := kgs.generateEmbedding(req.EmbeddingText)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to generate embedding: %v", err), http.StatusInternalServerError)
		return
	}
	node := Node{
		ID:       req.ID,
		Label:    req.Label,
		Type:     req.Type,
		Vector:   vector,
		Metadata: req.Metadata,
		Created:  time.Now(),
		Updated:  time.Now(),
	}
	kgs.nodes.Set(req.ID, node)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(node)
}

func (kgs *KnowledgeGraphServer) GetNode(w http.ResponseWriter, r *http.Request) {
	nodeID := chi.URLParam(r, "id")
	value, exists := kgs.nodes.Get(nodeID)
	if !exists {
		http.Error(w, "Node not found", http.StatusNotFound)
		return
	}
	node := value
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(node)
}

func (kgs *KnowledgeGraphServer) UpdateNode(w http.ResponseWriter, r *http.Request) {
	nodeID := chi.URLParam(r, "id")
	value, exists := kgs.nodes.Get(nodeID)
	if !exists {
		http.Error(w, "Node not found", http.StatusNotFound)
		return
	}
	node := value
	var req struct {
		Label        string                 `json:"label"`
		Type         string                 `json:"type"`
		EmbeddingText string                `json:"embedding_text"`
		Metadata     map[string]interface{} `json:"metadata"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.Label != "" {
		node.Label = req.Label
	}
	if req.Type != "" {
		node.Type = req.Type
	}
	if req.EmbeddingText != "" {
		vector, err := kgs.generateEmbedding(req.EmbeddingText)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to generate embedding: %v", err), http.StatusInternalServerError)
			return
		}
		node.Vector = vector
	}
	if req.Metadata != nil {
		node.Metadata = req.Metadata
	}
	node.Updated = time.Now()
	kgs.nodes.Set(nodeID, node)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(node)
}

func (kgs *KnowledgeGraphServer) DeleteNode(w http.ResponseWriter, r *http.Request) {
	nodeID := chi.URLParam(r, "id")
	kgs.nodes.Delete(nodeID)
	links := kgs.links.GetOrderedV2()
	for _, link := range links {
		link := link.Value
		if link.Source == nodeID || link.Target == nodeID {
			kgs.links.Delete(nodeID)
		}
	}
	w.WriteHeader(http.StatusNoContent)
}

func (kgs *KnowledgeGraphServer) AddRelation(w http.ResponseWriter, r *http.Request) {
	var link Link
	if err := json.NewDecoder(r.Body).Decode(&link); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if _, exists := kgs.nodes.Get(link.Source); !exists {
		http.Error(w, "Source node not found", http.StatusBadRequest)
		return
	}
	if _, exists := kgs.nodes.Get(link.Target); !exists {
		http.Error(w, "Target node not found", http.StatusBadRequest)
		return
	}
	linkID := fmt.Sprintf("%s-%s-%s", link.Source, link.Label, link.Target)
	kgs.links.Set(linkID, link)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(link)
}

func (kgs *KnowledgeGraphServer) GetGraph(w http.ResponseWriter, r *http.Request) {
	graph := Graph{
		Nodes: make([]Node, 0),
		Links: make([]Link, 0),
	}
	nodes := kgs.nodes.GetOrderedV2()
	for _, node := range nodes {
		graph.Nodes = append(graph.Nodes, node.Value)
	}
	links := kgs.links.GetOrderedV2()
	for _, link := range links {
		graph.Links = append(graph.Links, link.Value)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(graph)
}

func (kgs *KnowledgeGraphServer) SemanticSearch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Query string `json:"query"`
		K     int    `json:"k"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.K <= 0 {
		req.K = 5
	}
	queryVector, err := kgs.generateEmbedding(req.Query)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to generate embedding: %v", err), http.StatusInternalServerError)
		return
	}
	var results []SearchResult
	nodes := kgs.nodes.GetOrderedV2()
	for _, node := range nodes {
		node := node.Value
		similarity := cosineSimilarity(queryVector, node.Vector)
		results = append(results, SearchResult{
			Node:       node,
			Similarity: similarity,
		})
	}
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Similarity > results[i].Similarity {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
	if len(results) > req.K {
		results = results[:req.K]
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

func (kgs *KnowledgeGraphServer) FindSimilarNodes(w http.ResponseWriter, r *http.Request) {
	nodeID := chi.URLParam(r, "id")
	value, exists := kgs.nodes.Get(nodeID)
	if !exists {
		http.Error(w, "Node not found", http.StatusNotFound)
		return
	}
	sourceNode := value
	var results []SearchResult
	nodes := kgs.nodes.GetOrderedV2()
	for _, node := range nodes {
		node := node.Value
		if node.ID == nodeID {
			continue
		}
		similarity := cosineSimilarity(sourceNode.Vector, node.Vector)
		results = append(results, SearchResult{
			Node:       node,
			Similarity: similarity,
		})
	}
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Similarity > results[i].Similarity {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
	if len(results) > 5 {
		results = results[:5]
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

func (kgs *KnowledgeGraphServer) BatchImport(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Nodes []struct {
			ID           string                 `json:"id"`
			Label        string                 `json:"label"`
			Type         string                 `json:"type"`
			EmbeddingText string                `json:"embedding_text"`
			Metadata     map[string]interface{} `json:"metadata"`
		} `json:"nodes"`
		Links []Link `json:"links"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	for _, n := range req.Nodes {
		vector, _ := kgs.generateEmbedding(n.EmbeddingText)
		node := Node{
			ID:       n.ID,
			Label:    n.Label,
			Type:     n.Type,
			Vector:   vector,
			Metadata: n.Metadata,
			Created:  time.Now(),
			Updated:  time.Now(),
		}
		kgs.nodes.Set(n.ID, node)
	}
	for _, link := range req.Links {
		linkID := fmt.Sprintf("%s-%s-%s", link.Source, link.Label, link.Target)
		kgs.links.Set(linkID, link)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"nodes_imported": len(req.Nodes),
		"links_imported": len(req.Links),
	})
}

func (kgs *KnowledgeGraphServer) GetStatistics(w http.ResponseWriter, r *http.Request) {
	nodeCount := 0
	linkCount := 0
	typeDistribution := make(map[string]int)
	nodes := kgs.nodes.GetOrderedV2()
	for _, node := range nodes {
		nodeCount++
		node := node.Value
		typeDistribution[node.Type]++
	}
	links := kgs.links.GetOrderedV2()
	for _, link := range links {
		log.Println(link.Value)
		linkCount++
	}
	stats := map[string]interface{}{
		"node_count":        nodeCount,
		"link_count":        linkCount,
		"type_distribution": typeDistribution,
		"vector_dimension":  kgs.vectorDim,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func StartServer() {
	server, err := NewKnowledgeGraphServer()
	if err != nil {
		log.Fatalf("Failed to initialize server: %v", err)
	}
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Timeout(60 * time.Second))
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
		ExposedHeaders:   []string{"Link"},
		AllowCredentials: true,
		MaxAge:          300,
	}))
	r.Route("/api", func(r chi.Router) {
		r.Post("/nodes", server.AddNode)
		r.Get("/nodes/{id}", server.GetNode)
		r.Put("/nodes/{id}", server.UpdateNode)
		r.Delete("/nodes/{id}", server.DeleteNode)
		r.Post("/relations", server.AddRelation)
		r.Get("/graph", server.GetGraph)
		r.Post("/batch", server.BatchImport)
		r.Post("/search", server.SemanticSearch)
		r.Get("/similar/{id}", server.FindSimilarNodes)
		r.Get("/stats", server.GetStatistics)
	})
	r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
	})
	port := ":8080"
	log.Printf("Knowledge Graph Server starting on %s", port)
	if err := http.ListenAndServe(port, r); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}