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

func (a *Tensor) Stack(b *Tensor, axis int) (interface{}, error) {
	if a == nil || a.Data == nil || b == nil || b.Data == nil {
		return nil, errors.New("both tensors should be initialized")
	}

	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, errors.New("matrix is not compatible for the Stack operation")
	}

	if axis < 0 || axis > 2 {
		return nil, errors.New("axis must be between 0, 1 and 2")
	}

	switch axis {
	case 0:
		result := make([][][]float64, 2)
		result[0] = a.Data
		result[1] = b.Data
		return result, nil
	case 1:
		result := make([][][]float64, a.Rows)
		for i := 0; i < a.Rows; i++ {
			result[i] = make([][]float64, a.Cols)
			for j := 0; j < a.Cols; j++ {
				result[i][j] = make([]float64, 2)
				result[i][j][0] = a.Data[i][j]
				result[i][j][1] = b.Data[i][j]
			}
		}
		return result, nil
	default:
		return nil, errors.New("unexpected axis")
	}
}

func (t *Tensor) Split(axis int, parts int) ([]*Tensor, error) {
	if t == nil || t.Data == nil {
		return nil, errors.New("tensor is nil")
	}

	if axis == 0 {
		if t.Rows%parts != 0 {
			return nil, errors.New("rows cannot be evenly split")
		}
		step := t.Rows / parts
		result := make([]*Tensor, parts)

		for i := 0; i < parts; i++ {
			chunk := make([][]float64, step)
			for j := 0; j < step; j++ {
				chunk[j] = make([]float64, t.Cols)
				copy(chunk[j], t.Data[i*step+j])
			}
			result[i] = &Tensor{Data: chunk, Rows: step, Cols: t.Cols}
		}
		return result, nil

	} else if axis == 1 {
		if t.Cols%parts != 0 {
			return nil, errors.New("columns cannot be evenly split")
		}
		step := t.Cols / parts
		result := make([]*Tensor, parts)

		for i := 0; i < parts; i++ {
			chunk := make([][]float64, t.Rows)
			for j := 0; j < t.Rows; j++ {
				chunk[j] = make([]float64, step)
				copy(chunk[j], t.Data[j][i*step:(i+1)*step])
			}
			result[i] = &Tensor{Data: chunk, Rows: t.Rows, Cols: step}
		}
		return result, nil

	} else {
		return nil, errors.New("invalid axis; use 0 (rows) or 1 (cols)")
	}
}
