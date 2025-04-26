package Tensor

import (
	"errors"
	"log"
)

func (t *Tensor) Reshape(row, col int) (*Tensor, error) {
	if t == nil || t.Data == nil {
		return nil, errors.New("can not reshape uninitialized or empty array")
	}

	if row*col != t.Rows*t.Cols {
		return nil, errors.New("dimension does not matching")
	}

	result := make([][]float64, row)

	temp, err := t.Flatten()
	if err != nil {
		log.Fatalf("error in flatning the matrix : %v", err)
	}

	index := 0
	for i := 0; i < row; i++ {
		result[i] = make([]float64, col)
		for j := 0; j < col; j++ {
			result[i][j] = temp[index]
			index++
		}
	}

	return NewTensor(result)
}

//func (t *Tensor) ExpandDims()
