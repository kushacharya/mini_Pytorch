package tensor

import (
	"errors"
	"math"
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

func RandomNormal(row, col int, mean, sd float64) (*Tensor, error) {
	if row <= 0 || col <= 0 {
		return nil, errors.New("rows or columns can not be 0")
	}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	result := make([][]float64, row)
	for i := 0; i < row; i++ {
		result[i] = make([]float64, col)
		for j := 0; j < col; j++ {
			u1 := r.Float64()
			u2 := r.Float64()
			z := math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2*math.Pi*u2)
			result[i][j] = z*sd + mean
		}
	}
	return NewTensor(result)
}
