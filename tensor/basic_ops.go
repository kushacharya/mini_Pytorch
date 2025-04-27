package tensor

import (
	"errors"
	"fmt"
)

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

func (a *Tensor) AddScalar(Scalar float64) (*Tensor, error) {
	//var Scalar float64
	result := make([][]float64, a.Rows)
	for i := 0; i < a.Rows; i++ {
		result[i] = make([]float64, a.Cols)
		for j := 0; j < a.Cols; j++ {
			result[i][j] = a.Data[i][j] + Scalar
		}
	}
	t, err := NewTensor(result)
	if err != nil {
		return nil, fmt.Errorf("error in adding the scalar value to the matrix: %v", err)
	}
	return t, nil
}

func (a *Tensor) Transpose() (*Tensor, error) {
	result := make([][]float64, a.Cols)
	for i := 0; i < a.Cols; i++ {
		result[i] = make([]float64, a.Rows)
		for j := 0; j < a.Rows; j++ {
			result[i][j] = a.Data[j][i]
		}
	}
	t, err := NewTensor(result)
	if err != nil {
		return nil, fmt.Errorf("error during the transposing of matrix : %v", err)
	}
	return t, nil
}
