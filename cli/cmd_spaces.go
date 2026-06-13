package cli

import (
	"github.com/spf13/cobra"
	"github.com/tamnd/huggingface-cli/huggingface"
)

func (a *App) spacesCmd() *cobra.Command {
	var search string
	cmd := &cobra.Command{
		Use:   "spaces",
		Short: "List top liked spaces",
		RunE: func(cmd *cobra.Command, _ []string) error {
			n := a.effectiveLimit(20)
			a.progressf("fetching %d spaces...", n)
			spaces, err := a.client.Spaces(cmd.Context(), search, n)
			if err != nil {
				return mapFetchErr(err)
			}
			return a.renderOrEmpty(spaces, len(spaces))
		},
	}
	cmd.Flags().StringVar(&search, "search", "", "keyword search")
	return cmd
}

func (a *App) spaceCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "space <id>",
		Short: "Show a Space's detail",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			id := args[0]
			a.progressf("fetching space %q...", id)
			s, err := a.client.SpaceDetail(cmd.Context(), id)
			if err != nil {
				return mapFetchErr(err)
			}
			return a.render([]huggingface.Space{s})
		},
	}
}
