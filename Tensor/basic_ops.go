package Tensor

import "errors"

func (a *Tensor) Add(b *Tensor) (*Tensor, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, errors.New("tensor dimension does not match")
	}

	result := make([][]float64, a.Rows)
	for i := 0; i < a.Rows; i++ {
		result[i] = make([]float64, a.Cols)
		for j := 0; j < a.Cols; j++ {
			result[i][j] = a.Data[i][j] + b.Data[i][j]
		}
	}
	return NewTensor(result)
}

func (a *Tensor) Sub(b *Tensor) (*Tensor, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, errors.New("tensor dimension does not match")
	}

	result := make([][]float64, a.Rows)
	for i := 0; i < a.Rows; i++ {
		result[i] = make([]float64, a.Cols)
		for j := 0; j < a.Cols; j++ {
			result[i][j] = a.Data[i][j] - b.Data[i][j]
		}
	}
	return NewTensor(result)
}

func (a *Tensor) Multiply(b *Tensor) (*Tensor, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, errors.New("tensor dimension does not match")
	}

	result := make([][]float64, a.Rows)
	for i := 0; i < a.Rows; i++ {
		result[i] = make([]float64, a.Cols)
		for j := 0; j < a.Cols; j++ {
			result[i][j] = a.Data[i][j] * b.Data[i][j]
		}
	}

	return NewTensor(result)
}

func (a *Tensor) AddScalar(scalar float64) *Tensor {
	result := make([][]float64, a.Rows)
	for i := 0; i < a.Rows; i++ {
		result[i] = make([]float64, a.Cols)
		for j := 0; j < a.Cols; j++ {
			result[i][j] = a.Data[i][j] + scalar
		}
	}
	t, _ := NewTensor(result)
	return t
}
