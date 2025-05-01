package tensor

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

func (t *Tensor) ExpandDims(axis int) (*Tensor, error) {
	if t == nil || t.Data == nil {
		return nil, errors.New("can not do Expand Dimension for uninitialized tensor")
	}

	if axis < 0 || axis > 2 {
		return nil, errors.New("invalid axis")
	}

	switch axis {
	case 0:
		newData := make([][]float64, 1)
		for i := range t.Data {
			newData = append(newData, t.Data[i])
		}
		return NewTensor(newData)
	case 1:
		newData := make([][]float64, t.Rows)
		for i := 0; i < t.Rows; i++ {
			newData[i] = make([]float64, 1*t.Rows)
			copy(newData[i], t.Data[i])
		}
		return NewTensor(newData)
	case 2:
		newData := make([][]float64, t.Rows)
		for i := 0; i < t.Rows; i++ {
			newData[i] = make([]float64, t.Cols)
			for j := 0; j < t.Cols; j++ {
				newData[i][j] = t.Data[i][j]
			}
		}
		return NewTensor(newData)
	}

	return nil, errors.New("expected error")

}

func (t *Tensor) Squeeze(axis int) (*Tensor, error) {
	if t == nil || t.Data == nil {
		return nil, errors.New("can not squeeze the matrix that is uninitialized")
	}
	switch axis {
	case 0:
		if t.Rows == 1 {
			return NewTensor([][]float64{t.Data[0]})
		} else {
			return nil, errors.New("can not squeeze axis 0, size is not 1")
		}
	case 1:
		if t.Cols == 1 {
			flatten := make([][]float64, t.Rows)
			for i := 0; i < t.Rows; i++ {
				flatten[i] = []float64{t.Data[i][0]}
			}
			return NewTensor(flatten)
		} else {
			return nil, errors.New("can not squeeze axis 1, size is not 1")
		}
	default:
		return nil, errors.New("axis out of range")
	}
}
