package Tensor

import (
	"errors"
	"math/rand"
	"time"
)

func Zeros(rows, cols int) (*Tensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, errors.New("please enter rows and cols in positive integer")
	}
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = 0.0
		}
	}
	return NewTensor(result)
}

func Ones(rows, cols int) (*Tensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, errors.New("please enter rows and col in positive integer")
	}

	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = 1
		}
	}
	return NewTensor(result)
}

func Random(rows, cols int) (*Tensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, errors.New("please enter rows and col in positive integer")
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = r.Float64()
		}
	}
	return NewTensor(result)
}

func RandomInRange(rows, cols int, min, max float64) (*Tensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, errors.New("please enter rows and col in positive integer")
	}

	if min >= max {
		return nil, errors.New("for range the min can not exceed or equal the max value")
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = r.Float64() * (max - min)
		}
	}

	return NewTensor(result)
}
