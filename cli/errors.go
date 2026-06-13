package cli

import (
	"errors"

	"github.com/tamnd/huggingface-cli/huggingface"
)

func isNotFound(err error) bool {
	return errors.Is(err, huggingface.ErrNotFound)
}
