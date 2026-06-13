// Package huggingface is the library behind the hf command: the HTTP client,
// request shaping, and typed data models for the Hugging Face Hub.
//
// The Hub REST API at https://huggingface.co/api is open for public content
// with no authentication required. The Client sets a real User-Agent, paces
// requests, and retries 429/5xx responses with exponential backoff.
package huggingface

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sync"
	"time"
)

const defaultAPIBase = "https://huggingface.co/api"

// DefaultUserAgent identifies the client to the Hub.
const DefaultUserAgent = "hf/dev (+https://github.com/tamnd/huggingface-cli)"

// ErrNotFound is returned when the API returns HTTP 404.
var ErrNotFound = errors.New("not found")

// Config holds constructor parameters.
type Config struct {
	BaseURL   string
	UserAgent string
	Rate      time.Duration
	Retries   int
	Workers   int
	Timeout   time.Duration
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		BaseURL:   defaultAPIBase,
		UserAgent: DefaultUserAgent,
		Rate:      100 * time.Millisecond,
		Retries:   3,
		Workers:   8,
		Timeout:   30 * time.Second,
	}
}

// Client talks to the Hugging Face Hub API.
type Client struct {
	httpClient *http.Client
	baseURL    string
	userAgent  string
	rate       time.Duration
	retries    int
	workers    int
	mu         sync.Mutex
	last       time.Time
}

// NewClient returns a Client using cfg.
func NewClient(cfg Config) *Client {
	base := cfg.BaseURL
	if base == "" {
		base = defaultAPIBase
	}
	return &Client{
		httpClient: &http.Client{Timeout: cfg.Timeout},
		baseURL:    base,
		userAgent:  cfg.UserAgent,
		rate:       cfg.Rate,
		retries:    cfg.Retries,
		workers:    cfg.Workers,
	}
}

// get fetches a URL with pacing and retries.
func (c *Client) get(ctx context.Context, rawURL string) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt <= c.retries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff(attempt)):
			}
		}
		body, retry, err := c.do(ctx, rawURL)
		if err == nil {
			return body, nil
		}
		lastErr = err
		if !retry {
			return nil, err
		}
	}
	return nil, fmt.Errorf("get %s: %w", rawURL, lastErr)
}

func (c *Client) do(ctx context.Context, rawURL string) ([]byte, bool, error) {
	c.pace()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, false, err
	}
	req.Header.Set("User-Agent", c.userAgent)
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, true, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode == http.StatusNotFound {
		return nil, false, ErrNotFound
	}
	if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
		return nil, true, fmt.Errorf("http %d", resp.StatusCode)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, false, fmt.Errorf("http %d", resp.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, true, err
	}
	return b, false, nil
}

func (c *Client) pace() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.rate <= 0 {
		return
	}
	if wait := c.rate - time.Since(c.last); wait > 0 {
		time.Sleep(wait)
	}
	c.last = time.Now()
}

func backoff(attempt int) time.Duration {
	d := time.Duration(attempt) * 500 * time.Millisecond
	if d > 5*time.Second {
		d = 5 * time.Second
	}
	return d
}

// getJSON fetches and JSON-decodes into v.
func (c *Client) getJSON(ctx context.Context, rawURL string, v any) error {
	body, err := c.get(ctx, rawURL)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(body, v); err != nil {
		return fmt.Errorf("decode %s: %w", rawURL, err)
	}
	return nil
}

// ─── public methods ───────────────────────────────────────────────────────────

// Models returns the top-downloaded models, optionally filtered by query and
// task (pipeline_tag). limit specifies the max results to fetch.
func (c *Client) Models(ctx context.Context, query, task string, limit int) ([]Model, error) {
	if limit <= 0 {
		limit = 20
	}
	params := url.Values{}
	params.Set("sort", "downloads")
	params.Set("direction", "-1")
	params.Set("limit", fmt.Sprintf("%d", limit))
	if query != "" {
		params.Set("search", query)
	}
	if task != "" {
		params.Set("filter", task)
	}
	rawURL := c.baseURL + "/models?" + params.Encode()

	var raw []apiModel
	if err := c.getJSON(ctx, rawURL, &raw); err != nil {
		return nil, err
	}
	out := make([]Model, 0, len(raw))
	for i, m := range raw {
		m := m
		out = append(out, apiModelToModel(&m, i+1))
	}
	return out, nil
}

// ModelDetail fetches a single model by id (e.g. "meta-llama/Llama-2-7b-hf").
func (c *Client) ModelDetail(ctx context.Context, id string) (Model, error) {
	// HF ids contain a slash; use the raw id to preserve the path.
	rawURL := c.baseURL + "/models/" + id
	var raw apiModel
	if err := c.getJSON(ctx, rawURL, &raw); err != nil {
		return Model{}, err
	}
	return apiModelToModel(&raw, 1), nil
}

// Datasets returns the top-downloaded datasets, optionally filtered by query.
func (c *Client) Datasets(ctx context.Context, query string, limit int) ([]Dataset, error) {
	if limit <= 0 {
		limit = 20
	}
	params := url.Values{}
	params.Set("sort", "downloads")
	params.Set("direction", "-1")
	params.Set("limit", fmt.Sprintf("%d", limit))
	if query != "" {
		params.Set("search", query)
	}
	rawURL := c.baseURL + "/datasets?" + params.Encode()

	var raw []apiDataset
	if err := c.getJSON(ctx, rawURL, &raw); err != nil {
		return nil, err
	}
	out := make([]Dataset, 0, len(raw))
	for i, d := range raw {
		d := d
		out = append(out, apiDatasetToDataset(&d, i+1))
	}
	return out, nil
}

// DatasetDetail fetches a single dataset by id.
func (c *Client) DatasetDetail(ctx context.Context, id string) (Dataset, error) {
	rawURL := c.baseURL + "/datasets/" + id
	var raw apiDataset
	if err := c.getJSON(ctx, rawURL, &raw); err != nil {
		return Dataset{}, err
	}
	return apiDatasetToDataset(&raw, 1), nil
}

// Spaces returns the top-liked Spaces, optionally filtered by query.
func (c *Client) Spaces(ctx context.Context, query string, limit int) ([]Space, error) {
	if limit <= 0 {
		limit = 20
	}
	params := url.Values{}
	params.Set("sort", "likes")
	params.Set("direction", "-1")
	params.Set("limit", fmt.Sprintf("%d", limit))
	if query != "" {
		params.Set("search", query)
	}
	rawURL := c.baseURL + "/spaces?" + params.Encode()

	var raw []apiSpace
	if err := c.getJSON(ctx, rawURL, &raw); err != nil {
		return nil, err
	}
	out := make([]Space, 0, len(raw))
	for i, s := range raw {
		s := s
		out = append(out, apiSpaceToSpace(&s, i+1))
	}
	return out, nil
}

// SpaceDetail fetches a single Space by id.
func (c *Client) SpaceDetail(ctx context.Context, id string) (Space, error) {
	rawURL := c.baseURL + "/spaces/" + id
	var raw apiSpace
	if err := c.getJSON(ctx, rawURL, &raw); err != nil {
		return Space{}, err
	}
	return apiSpaceToSpace(&raw, 1), nil
}

// Papers returns the daily papers feed, optionally filtered by query.
func (c *Client) Papers(ctx context.Context, query string, limit int) ([]Paper, error) {
	if limit <= 0 {
		limit = 20
	}
	params := url.Values{}
	params.Set("limit", fmt.Sprintf("%d", limit))
	if query != "" {
		params.Set("q", query)
	}
	rawURL := c.baseURL + "/papers?" + params.Encode()

	var raw []apiPaper
	if err := c.getJSON(ctx, rawURL, &raw); err != nil {
		return nil, err
	}
	if limit > 0 && len(raw) > limit {
		raw = raw[:limit]
	}
	out := make([]Paper, 0, len(raw))
	for i, p := range raw {
		p := p
		out = append(out, apiPaperToPaper(&p, i+1))
	}
	return out, nil
}
