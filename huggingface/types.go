package huggingface

import (
	"strings"
	"time"
)

// Model is the record emitted for Hugging Face model results.
type Model struct {
	Rank      int    `json:"rank"`
	ID        string `json:"id"`
	Task      string `json:"task"`
	Downloads int64  `json:"downloads"`
	Likes     int    `json:"likes"`
	Tags      string `json:"tags"`
	Updated   string `json:"updated"`
	URL       string `json:"url"`
}

// Dataset is the record emitted for Hugging Face dataset results.
type Dataset struct {
	Rank      int    `json:"rank"`
	ID        string `json:"id"`
	Downloads int64  `json:"downloads"`
	Likes     int    `json:"likes"`
	Tags      string `json:"tags"`
	Updated   string `json:"updated"`
	URL       string `json:"url"`
}

// Space is the record emitted for Hugging Face Spaces results.
type Space struct {
	Rank    int    `json:"rank"`
	ID      string `json:"id"`
	SDK     string `json:"sdk"`
	Likes   int    `json:"likes"`
	Tags    string `json:"tags"`
	Updated string `json:"updated"`
	URL     string `json:"url"`
}

// Paper is the record emitted for Hugging Face daily papers results.
type Paper struct {
	Rank      int    `json:"rank"`
	ID        string `json:"id"`
	Title     string `json:"title"`
	Authors   string `json:"authors"`
	Upvotes   int    `json:"upvotes"`
	Published string `json:"published"`
	URL       string `json:"url"`
}

// ─── wire types from the HF Hub API ─────────────────────────────────────────

type apiModel struct {
	ID           string   `json:"id"`
	ModelID      string   `json:"modelId"`
	Author       string   `json:"author"`
	LastModified string   `json:"lastModified"`
	Downloads    int64    `json:"downloads"`
	Likes        int      `json:"likes"`
	PipelineTag  string   `json:"pipeline_tag"`
	Tags         []string `json:"tags"`
}

type apiDataset struct {
	ID           string   `json:"id"`
	Author       string   `json:"author"`
	LastModified string   `json:"lastModified"`
	Downloads    int64    `json:"downloads"`
	Likes        int      `json:"likes"`
	Tags         []string `json:"tags"`
}

type apiSpace struct {
	ID           string   `json:"id"`
	Author       string   `json:"author"`
	LastModified string   `json:"lastModified"`
	Likes        int      `json:"likes"`
	Tags         []string `json:"tags"`
	SDK          string   `json:"sdk"`
}

type apiPaper struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Abstract    string `json:"abstract"`
	Upvotes     int    `json:"upvotes"`
	PublishedAt string `json:"publishedAt"`
	Paper       struct {
		Authors []struct {
			Name string `json:"name"`
		} `json:"authors"`
	} `json:"paper"`
}

// ─── URL helpers ─────────────────────────────────────────────────────────────

func hfModelURL(id string) string {
	return "https://huggingface.co/" + id
}

func hfDatasetURL(id string) string {
	return "https://huggingface.co/datasets/" + id
}

func hfSpaceURL(id string) string {
	return "https://huggingface.co/spaces/" + id
}

func hfPaperURL(id string) string {
	return "https://huggingface.co/papers/" + id
}

// ─── field helpers ────────────────────────────────────────────────────────────

// parseDateString parses an ISO-8601 timestamp and formats it as "2006-01-02".
// Returns "" on parse failure.
func parseDateString(s string) string {
	if s == "" {
		return ""
	}
	formats := []string{time.RFC3339, time.RFC3339Nano, "2006-01-02T15:04:05.999Z", "2006-01-02"}
	for _, f := range formats {
		if t, err := time.Parse(f, s); err == nil {
			return t.UTC().Format("2006-01-02")
		}
	}
	if len(s) >= 10 {
		return s[:10]
	}
	return s
}

// firstTags joins the first n tags with ", ". Returns "" when tags is empty.
func firstTags(tags []string, n int) string {
	if len(tags) == 0 {
		return ""
	}
	if n > len(tags) {
		n = len(tags)
	}
	return strings.Join(tags[:n], ", ")
}

// authorNames joins all author names with ", " and truncates to maxLen chars.
func authorNames(names []string, maxLen int) string {
	s := strings.Join(names, ", ")
	if maxLen > 0 && len(s) > maxLen {
		return s[:maxLen-3] + "..."
	}
	return s
}

// extractAuthorNames pulls Name fields from the paper's authors slice.
func extractAuthorNames(authors []struct {
	Name string `json:"name"`
}) []string {
	names := make([]string, 0, len(authors))
	for _, a := range authors {
		if a.Name != "" {
			names = append(names, a.Name)
		}
	}
	return names
}

// ─── wire-to-record converters ───────────────────────────────────────────────

func apiModelToModel(m *apiModel, rank int) Model {
	id := m.ID
	if id == "" {
		id = m.ModelID
	}
	return Model{
		Rank:      rank,
		ID:        id,
		Task:      m.PipelineTag,
		Downloads: m.Downloads,
		Likes:     m.Likes,
		Tags:      firstTags(m.Tags, 3),
		Updated:   parseDateString(m.LastModified),
		URL:       hfModelURL(id),
	}
}

func apiDatasetToDataset(d *apiDataset, rank int) Dataset {
	return Dataset{
		Rank:      rank,
		ID:        d.ID,
		Downloads: d.Downloads,
		Likes:     d.Likes,
		Tags:      firstTags(d.Tags, 3),
		Updated:   parseDateString(d.LastModified),
		URL:       hfDatasetURL(d.ID),
	}
}

func apiSpaceToSpace(s *apiSpace, rank int) Space {
	return Space{
		Rank:    rank,
		ID:      s.ID,
		SDK:     s.SDK,
		Likes:   s.Likes,
		Tags:    firstTags(s.Tags, 3),
		Updated: parseDateString(s.LastModified),
		URL:     hfSpaceURL(s.ID),
	}
}

func apiPaperToPaper(p *apiPaper, rank int) Paper {
	return Paper{
		Rank:      rank,
		ID:        p.ID,
		Title:     p.Title,
		Authors:   authorNames(extractAuthorNames(p.Paper.Authors), 80),
		Upvotes:   p.Upvotes,
		Published: parseDateString(p.PublishedAt),
		URL:       hfPaperURL(p.ID),
	}
}
