package tensor

import (
	"errors"
	"fmt"
)

func (t *Tensor) Get(row, col int) (float64, error) {
	if t == nil || t.Data == nil {
		return 0.0, errors.New("can not access the element of uninitialized tensor")
	}
	if row < 0 || col < 0 || row > t.Rows || col > t.Cols {
		return 0.0, errors.New("can not access the element that is out of the range of tensor")
	}

	return t.Data[row][col], nil
}

func (t *Tensor) Set(row, col int, value float64) (bool, error) {
	if t == nil || t.Data == nil {
		return false, errors.New("can not access the element of uninitialized tensor")
	}
	if row < 0 || col < 0 || row > t.Rows || col > t.Cols {
		return false, errors.New("can not access the element that is out of the range of tensor")
	}

	t.Data[row][col] = value
	fmt.Printf("value updated")
	return true, nil

}

func (t *Tensor) Slice(rowStart, rowEnd, colStart, colEnd int) (*Tensor, error) {
	if t == nil || t.Data == nil {
		return nil, errors.New("can not slice an uninitialized tensor")
	}
	if rowStart < 0 || rowEnd < 0 || colStart < 0 || colEnd < 0 {
		return nil, errors.New("can not access negative column in tensor")
	}

	result := make([][]float64, rowEnd-rowStart)
	for i := rowStart; i <= rowEnd; i++ {
		result[i] = make([]float64, colEnd-colStart)
		for j := colStart; j <= colEnd; j++ {
			result[i][j] = t.Data[i][j]
		}
	}

	return NewTensor(result)
}
