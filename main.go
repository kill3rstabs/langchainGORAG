package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log"
	"net/url"
	"os"
	"strings"
	"sync"

	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

const maxContextLength = 5 // Number of previous exchanges to keep in context

type Message struct {
	Msg string `json:"msg"`
}

type PromptTemplate struct {
	SystemMessage      string
	ContextFormat      string
	RelevantInfoFormat string
	UserQueryFormat    string
}

var defaultPromptTemplate = PromptTemplate{
	SystemMessage:      "You are a helpful AI assistant. Provide concise and accurate responses based on the given context and relevant information.Only answer from the given data donot answer from anywhere else or your prior memory. If you are unable to find the answer in the data then simply answer 'I don't know.' How can I assist you further?",
	ContextFormat:      "Previous conversation:\n%s\n",
	RelevantInfoFormat: "Relevant information:\n%s\n",
	// UserQueryFormat: "User Query: %s\nAssistant Response:",
}

type ChatContext struct {
	Context []string
	mu      sync.Mutex
}

var chatContext = ChatContext{
	Context: make([]string, 0, maxContextLength*2),
}

func chat(c *gin.Context) {
	var msg Message
	err := c.BindJSON(&msg)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}
	response := RAG(msg.Msg)
	c.JSON(201, gin.H{"message": response})
}

func main() {
	r := gin.New()
	r.POST("/chat", chat)
	r.Run(":8080")
}

func RAG(msg string) string {
	ctx := context.Background()
	collectionName := "rag"
	address := "http://localhost:6333"

	ollamaLLM, err := ollama.New(ollama.WithModel("llama3"))
	// ollamaLLM,err = ollama.makeOllamaOptionsFroTell the capitals of countries in europe ?mOptions(ollama.WithModel("llama3"))

	if err != nil {
		log.Fatal(err)
	}

	ollamaEmbedder, err := embeddings.NewEmbedder(ollamaLLM)
	if err != nil {
		log.Fatal(err)
	}

	url, err := url.Parse(address)
	if err != nil {
		log.Fatal(err)
	}

	store, err := qdrant.New(
		qdrant.WithURL(*url),
		qdrant.WithCollectionName(collectionName),
		qdrant.WithEmbedder(ollamaEmbedder),
	)
	if err != nil {
		log.Fatal(err)
	}

	err = createCollectionIfNotExists(address, collectionName)
	if err != nil {
		log.Fatal(err)
	}

	docs, err := readDocumentsFromCSV("healthcare_dataset.csv")
	if err != nil {
		log.Fatal(err)
	}

	_, err = store.AddDocuments(ctx, docs)
	if err != nil {
		log.Fatal(err)
	}

	chatContext.mu.Lock()
	chatContext.Context = append(chatContext.Context, "User: "+msg)
	chatContext.mu.Unlock()

	relevantDocs, err := store.SimilaritySearch(ctx, msg, 3)
	if err != nil {
		log.Printf("Error performing similarity search: %v", err)
	}

	prompt := constructPrompt(chatContext.Context, relevantDocs, msg, defaultPromptTemplate)

	response, err := ollamaLLM.Call(ctx, prompt)
	if err != nil {
		log.Printf("Error generating response: %v", err)
	}

	chatContext.mu.Lock()
	chatContext.Context = append(chatContext.Context, "Assistant: "+response)
	if len(chatContext.Context) > maxContextLength*2 {
		chatContext.Context = chatContext.Context[2:] // Remove oldest exchange
	}
	chatContext.mu.Unlock()

	return response
}

func readDocumentsFromCSV(filename string) ([]schema.Document, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var docs []schema.Document
	for _, record := range records {
		if len(record) < 1 {
			continue // Skip empty rows
		}
		doc := schema.Document{
			PageContent: record[0],
			Metadata:    map[string]any{"id": uuid.New().String()},
		}
		// If there are additional columns, add them as metadata
		for i := 1; i < len(record); i++ {
			doc.Metadata[fmt.Sprintf("column_%d", i)] = record[i]
		}
		docs = append(docs, doc)
	}

	return docs, nil
}

func createCollectionIfNotExists(address string, collectionName string) error {
	// Check if collection exists
	checkURL := fmt.Sprintf("%s/collections/%s", address, collectionName)
	resp, err := http.Get(checkURL)
	if err != nil {
		return fmt.Errorf("failed to check collection: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		log.Printf("Collection %s already exists", collectionName)
		return nil
	}

	// Collection doesn't exist, create it
	createURL := fmt.Sprintf("%s/collections/%s", address, collectionName)
	createReq := map[string]interface{}{
		"vectors": map[string]interface{}{
			"size":     4096,
			"distance": "Cosine",
		},
	}

	jsonData, err := json.Marshal(createReq)
	if err != nil {
		return fmt.Errorf("failed to marshal create request: %v", err)
	}

	req, err := http.NewRequest("PUT", createURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err = client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send create request: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to create collection: unexpected status code %d, body: %s", resp.StatusCode, string(body))
	}

	log.Printf("Created collection: %s", collectionName)
	return nil
}

func constructPrompt(context []string, relevantDocs []schema.Document, userQuery string, template PromptTemplate) string {
	var sb strings.Builder

	sb.WriteString(template.SystemMessage)
	sb.WriteString("\n\n")

	if len(context) > 0 {
		sb.WriteString(fmt.Sprintf(template.ContextFormat, strings.Join(context, "\n")))
		sb.WriteString("\n")
	}

	if len(relevantDocs) > 0 {
		var relevantInfo strings.Builder
		for _, doc := range relevantDocs {
			relevantInfo.WriteString(doc.PageContent)
			relevantInfo.WriteString("\n")
		}
		sb.WriteString(fmt.Sprintf(template.RelevantInfoFormat, relevantInfo.String()))
		sb.WriteString("\n")
	}

	sb.WriteString(fmt.Sprintf(template.UserQueryFormat, userQuery))

	return sb.String()
}
