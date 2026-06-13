package cli

import (
	"github.com/spf13/cobra"
	"github.com/tamnd/huggingface-cli/huggingface"
)

func (a *App) modelsCmd() *cobra.Command {
	var (
		task   string
		search string
	)
	cmd := &cobra.Command{
		Use:   "models",
		Short: "List top downloaded models",
		RunE: func(cmd *cobra.Command, _ []string) error {
			n := a.effectiveLimit(20)
			a.progressf("fetching %d models...", n)
			models, err := a.client.Models(cmd.Context(), search, task, n)
			if err != nil {
				return mapFetchErr(err)
			}
			return a.renderOrEmpty(models, len(models))
		},
	}
	cmd.Flags().StringVar(&task, "task", "", "filter by ML task (e.g. text-generation)")
	cmd.Flags().StringVar(&search, "search", "", "keyword search")
	return cmd
}

func (a *App) modelCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "model <id>",
		Short: "Show a model's detail card",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			id := args[0]
			a.progressf("fetching model %q...", id)
			m, err := a.client.ModelDetail(cmd.Context(), id)
			if err != nil {
				return mapFetchErr(err)
			}
			return a.render([]huggingface.Model{m})
		},
	}
}
