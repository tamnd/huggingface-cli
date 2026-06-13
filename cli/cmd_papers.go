package cli

import (
	"github.com/spf13/cobra"
)

func (a *App) papersCmd() *cobra.Command {
	var search string
	cmd := &cobra.Command{
		Use:   "papers",
		Short: "List daily papers from the Hub feed",
		RunE: func(cmd *cobra.Command, _ []string) error {
			n := a.effectiveLimit(20)
			a.progressf("fetching %d papers...", n)
			papers, err := a.client.Papers(cmd.Context(), search, n)
			if err != nil {
				return mapFetchErr(err)
			}
			return a.renderOrEmpty(papers, len(papers))
		},
	}
	cmd.Flags().StringVar(&search, "search", "", "keyword search")
	return cmd
}
