package tensor

import "errors"

func (a *Tensor) Concat(b *Tensor, axis int) (*Tensor, error) {
	if a == nil || a.Data == nil {
		return nil, errors.New("initialize the first tensor")
	}

	if axis != 0 && axis != 1 {
		return nil, errors.New("axis must be 0 (row-wise) or 1 (column-wise)")
	}

	if b == nil || b.Data == nil {
		return nil, errors.New("initialize the second tensor")
	}
	switch axis {
	case 0:
		if a.Cols != b.Cols {
			return nil, errors.New("can not concat tensors with different column sizes for axis 0")
		} else {
			result := make([][]float64, a.Rows+b.Rows)
			for i := 0; i < a.Rows; i++ {
				result[i] = make([]float64, a.Cols)
				copy(result[i], a.Data[i])
			}
			for i := 0; i < b.Rows; i++ {
				result[a.Rows+i] = make([]float64, b.Cols)
				copy(result[a.Rows+i], b.Data[i])
			}
			return NewTensor(result)
		}
	case 1:
		if a.Rows != b.Rows {
			return nil, errors.New("can not concat tensors with different column sizes for axis 1")
		} else {
			result := make([][]float64, a.Rows)
			for i := 0; i < a.Rows; i++ {
				result[i] = make([]float64, a.Cols+b.Cols)
				copy(result[i][:a.Cols], a.Data[i])
				copy(result[i][a.Cols:], b.Data[i])
			}

			return NewTensor(result)
		}
	default:
		return nil, errors.New("unexpected error occurred while concatenation")
	}
}
