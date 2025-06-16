package tensor

import (
	"errors"
)

func BroadcastShapes(ShapeA, ShapeB []int) ([]int, error) {
	result := []int{}
	lenA, lenB := len(ShapeA), len(ShapeB)
	maxLen := max(lenA, lenB)

	for i := 0; i < maxLen; i++ {
		dimA := getOrDefault(ShapeA, lenA-i-1, 1)
		dimB := getOrDefault(ShapeB, lenB-i-1, 1)

		if dimA == dimB || dimA == 1 || dimB == 1 {
			result = append([]int{max(dimA, dimB)}, result...)
		} else {
			return nil, errors.New("both the shapes can not be broadcasted")
		}
	}
	return result, nil
}

func getOrDefault(arr []int, idx, def int) int {
	if idx < 0 || idx >= len(arr) {
		return def
	}
	return arr[idx]
}
