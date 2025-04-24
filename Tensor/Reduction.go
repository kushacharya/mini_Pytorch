package Tensor

import (
	"errors"
	"math"
)

func (t *Tensor) Sum() (float64, error) {
	if t == nil || t.Data == nil {
		return 0, errors.New("to do sum the data must not be empty")
	}

	sum := 0.0

	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Cols; j++ {
			sum += t.Data[i][j]
		}
	}
	return sum, nil
}

func (t *Tensor) Mean() (float64, error) {
	if t == nil || t.Data == nil {
		return 0, errors.New("can not calculate mean of empty tensor")
	}

	mean := 0.0
	count := 0.0

	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Cols; j++ {
			mean += t.Data[i][j]
			count++
		}
	}

	mean = mean / count

	return mean, nil
}

func (t *Tensor) Max() (float64, error) {
	if t == nil || t.Data == nil {
		return 0, errors.New("can not find maximum of empty tensor")
	}

	tensorMax := math.Inf(-1)

	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Cols; j++ {
			tensorMax = math.Max(t.Data[i][j], tensorMax)
		}
	}

	return tensorMax, nil
}

func (t *Tensor) Min() (float64, error) {
	if t == nil || t.Data == nil {
		return 0, errors.New("can not find minimum of empty tensor")
	}

	tensorMin := math.Inf(+1)

	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Cols; j++ {
			tensorMin = math.Min(t.Data[i][j], tensorMin)
		}
	}

	return tensorMin, nil
}

func (t *Tensor) ArgMax() (int, int, error) {
	if t == nil || t.Data == nil {
		return 0, 0, errors.New("can not do ArgMax operation on null or uninitialized tensor")
	}

	tensorMax := math.Inf(-1)
	maxCol := 0
	maxRow := 0

	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Cols; j++ {
			if t.Data[i][j] > tensorMax {
				tensorMax = t.Data[i][j]
				maxRow = i
				maxCol = j
			}
		}
	}

	return maxRow, maxCol, nil

}
