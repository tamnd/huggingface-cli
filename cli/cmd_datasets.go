package cli

import (
	"github.com/spf13/cobra"
	"github.com/tamnd/huggingface-cli/huggingface"
)

func (a *App) datasetsCmd() *cobra.Command {
	var search string
	cmd := &cobra.Command{
		Use:   "datasets",
		Short: "List top downloaded datasets",
		RunE: func(cmd *cobra.Command, _ []string) error {
			n := a.effectiveLimit(20)
			a.progressf("fetching %d datasets...", n)
			datasets, err := a.client.Datasets(cmd.Context(), search, n)
			if err != nil {
				return mapFetchErr(err)
			}
			return a.renderOrEmpty(datasets, len(datasets))
		},
	}
	cmd.Flags().StringVar(&search, "search", "", "keyword search")
	return cmd
}

func (a *App) datasetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "dataset <id>",
		Short: "Show a dataset's detail card",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			id := args[0]
			a.progressf("fetching dataset %q...", id)
			d, err := a.client.DatasetDetail(cmd.Context(), id)
			if err != nil {
				return mapFetchErr(err)
			}
			return a.render([]huggingface.Dataset{d})
		},
	}
}
