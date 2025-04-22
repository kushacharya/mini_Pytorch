package Tensor

import (
	"errors"
	"fmt"
)

// Tensor this is like new data type that Tensor is new block
type Tensor struct {
	Data [][]float64
	Rows int
	Cols int
}

// NewTensor new Tensor is taking 2D matrix as input returns Tensor and error
func NewTensor(data [][]float64) (*Tensor, error) { // pointer to the Tensor and returns error
	if len(data) == 0 || len(data[0]) == 0 {
		return nil, errors.New("data can not be empty")

	}
	rows := len(data)
	cols := len(data[0])

	for _, row := range data {
		if len(row) != rows {
			return nil, errors.New(" all rows must have same number of columns")
		}
	}

	return &Tensor{
		Data: data,
		Rows: rows,
		Cols: cols,
	}, nil
}

// Shape created the shape finc that will give num of rows and cols for the matrix
func (t *Tensor) Shape() (int, int, error) {
	if t == nil || t.Data == nil {
		return 0, 0, errors.New("tensor is nil or uninitialized")
	}

	expectedCols := len(t.Data[0])
	for i := 0; i < len(t.Data); i++ {
		if len(t.Data[i]) != expectedCols {
			return 0, 0, errors.New("inconsistent Row or Column")
		}
	}

	return t.Rows, t.Cols, nil
}

// Clone creating the deep copy of the matrix
func (t *Tensor) Clone() (*Tensor, error) {
	if t == nil || t.Data == nil {
		return nil, errors.New("given tensor for clone is empty")
	}

	clone := make([][]float64, t.Rows)
	for i := 0; i < t.Rows; i++ {
		clone[i] = make([]float64, t.Cols)
		for j := 0; j < t.Cols; j++ {
			clone[i][j] = t.Data[i][j]
		}
	}

	return NewTensor(clone)
}

func (t *Tensor) Print() {
	for _, rows := range t.Data {
		fmt.Println(rows)
	}
}
