package huggingface_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/tamnd/huggingface-cli/huggingface"
)

func newTestClient(t *testing.T, mux *http.ServeMux) *huggingface.Client {
	t.Helper()
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	cfg := huggingface.DefaultConfig()
	cfg.BaseURL = ts.URL
	cfg.Rate = 0
	return huggingface.NewClient(cfg)
}

func encodeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func TestModels(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		encodeJSON(w, []map[string]any{
			{"id": "meta-llama/Llama-2-7b-hf", "pipeline_tag": "text-generation",
				"downloads": 100000, "likes": 500, "lastModified": "2024-01-15T00:00:00.000Z",
				"tags": []string{"pytorch", "llama"}},
		})
	})
	c := newTestClient(t, mux)
	models, err := c.Models(context.Background(), "", "", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 {
		t.Fatalf("got %d models, want 1", len(models))
	}
	if models[0].ID != "meta-llama/Llama-2-7b-hf" {
		t.Errorf("wrong ID: %s", models[0].ID)
	}
	if models[0].Task != "text-generation" {
		t.Errorf("wrong Task: %s", models[0].Task)
	}
	if models[0].Updated != "2024-01-15" {
		t.Errorf("wrong Updated: %s", models[0].Updated)
	}
}

func TestDatasets(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/datasets", func(w http.ResponseWriter, r *http.Request) {
		encodeJSON(w, []map[string]any{
			{"id": "openai/webtext", "downloads": 50000, "likes": 200,
				"lastModified": "2023-06-01T00:00:00.000Z", "tags": []string{"text", "en"}},
			{"id": "EleutherAI/pile", "downloads": 30000, "likes": 150,
				"lastModified": "2023-05-10T00:00:00.000Z", "tags": []string{"text"}},
		})
	})
	c := newTestClient(t, mux)
	datasets, err := c.Datasets(context.Background(), "", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(datasets) != 2 {
		t.Fatalf("got %d datasets, want 2", len(datasets))
	}
	if datasets[0].ID != "openai/webtext" {
		t.Errorf("wrong ID: %s", datasets[0].ID)
	}
	if datasets[0].Rank != 1 {
		t.Errorf("wrong rank: %d", datasets[0].Rank)
	}
	if datasets[1].Rank != 2 {
		t.Errorf("wrong rank for second dataset: %d", datasets[1].Rank)
	}
}

func TestSpaces(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/spaces", func(w http.ResponseWriter, r *http.Request) {
		encodeJSON(w, []map[string]any{
			{"id": "stabilityai/stable-diffusion", "sdk": "gradio", "likes": 8000,
				"lastModified": "2024-02-20T00:00:00.000Z", "tags": []string{"image-generation"}},
		})
	})
	c := newTestClient(t, mux)
	spaces, err := c.Spaces(context.Background(), "", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(spaces) != 1 {
		t.Fatalf("got %d spaces, want 1", len(spaces))
	}
	if spaces[0].ID != "stabilityai/stable-diffusion" {
		t.Errorf("wrong ID: %s", spaces[0].ID)
	}
	if spaces[0].SDK != "gradio" {
		t.Errorf("wrong SDK: %s", spaces[0].SDK)
	}
	if spaces[0].Likes != 8000 {
		t.Errorf("wrong Likes: %d", spaces[0].Likes)
	}
}

func TestPapers(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/papers", func(w http.ResponseWriter, r *http.Request) {
		encodeJSON(w, []map[string]any{
			{
				"id": "2302.13971", "title": "LLaMA: Open and Efficient Foundation Language Models",
				"upvotes": 1200, "publishedAt": "2023-02-24T00:00:00.000Z",
				"paper": map[string]any{
					"authors": []map[string]any{
						{"name": "Hugo Touvron"},
						{"name": "Thibaut Lavril"},
					},
				},
			},
		})
	})
	c := newTestClient(t, mux)
	papers, err := c.Papers(context.Background(), "", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(papers) != 1 {
		t.Fatalf("got %d papers, want 1", len(papers))
	}
	if papers[0].ID != "2302.13971" {
		t.Errorf("wrong ID: %s", papers[0].ID)
	}
	if papers[0].Title != "LLaMA: Open and Efficient Foundation Language Models" {
		t.Errorf("wrong Title: %s", papers[0].Title)
	}
	if papers[0].Authors != "Hugo Touvron, Thibaut Lavril" {
		t.Errorf("wrong Authors: %s", papers[0].Authors)
	}
	if papers[0].Published != "2023-02-24" {
		t.Errorf("wrong Published: %s", papers[0].Published)
	}
}

func TestModelDetail(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/models/meta-llama/Llama-2-7b-hf", func(w http.ResponseWriter, r *http.Request) {
		encodeJSON(w, map[string]any{
			"id": "meta-llama/Llama-2-7b-hf", "pipeline_tag": "text-generation",
			"downloads": 5000000, "likes": 12000, "lastModified": "2024-03-01T00:00:00.000Z",
			"tags": []string{"pytorch", "llama", "transformers"},
		})
	})
	c := newTestClient(t, mux)
	m, err := c.ModelDetail(context.Background(), "meta-llama/Llama-2-7b-hf")
	if err != nil {
		t.Fatal(err)
	}
	if m.ID != "meta-llama/Llama-2-7b-hf" {
		t.Errorf("wrong ID: %s", m.ID)
	}
	if m.Downloads != 5000000 {
		t.Errorf("wrong Downloads: %d", m.Downloads)
	}
}

func TestGetSendsUserAgent(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("User-Agent") == "" {
			t.Error("request carried no User-Agent")
		}
		encodeJSON(w, []map[string]any{})
	})
	c := newTestClient(t, mux)
	_, err := c.Models(context.Background(), "", "", 10)
	if err != nil {
		t.Fatal(err)
	}
}

func TestNotFound(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/models/nonexistent/model", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	})
	c := newTestClient(t, mux)
	_, err := c.ModelDetail(context.Background(), "nonexistent/model")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}
