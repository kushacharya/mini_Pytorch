package tensor

import (
	"errors"
)

func (a *Tensor) MatMul(b *Tensor) (*Tensor, error) {
	if a.Cols != b.Rows {
		return nil, errors.New("can not multiply : incompatible dimensions")
	}

	result := make([][]float64, a.Rows)
	for i := range result {
		result[i] = make([]float64, b.Cols)
	}

	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Cols; k++ {
				sum += a.Data[i][k] * b.Data[k][j]
			}
			result[i][j] = sum
		}
	}

	return NewTensor(result)

}
