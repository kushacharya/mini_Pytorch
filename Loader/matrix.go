package Loader

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func GetUserMatrix(name string) ([][]float64, error) {
	reader := bufio.NewReader(os.Stdin)

	fmt.Printf("Enter the number of Rows of Matrix for Tensor %s: ", name)
	rowStr, _ := reader.ReadString('\n')
	rows, err := strconv.Atoi(strings.TrimSpace(rowStr))
	if err != nil {
		return nil, fmt.Errorf("invalid Row input: %v", err)
	}

	fmt.Printf("Enter the number of the columns in tensor %s: ", name)
	colStr, _ := reader.ReadString('\n')
	cols, err := strconv.Atoi(strings.TrimSpace(colStr))
	if err != nil {
		return nil, fmt.Errorf("invalid column input: %v ", err)
	}

	matrix := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		fmt.Printf("Enter row %d values (space saperated): ", i+1)
		line, _ := reader.ReadString('\n')
		tokens := strings.Fields(line)

		if len(tokens) != cols {
			return nil, fmt.Errorf("expected %d values, got %d", cols, len(tokens))
		}

		row := make([]float64, cols)
		for j, tok := range tokens {
			val, err := strconv.ParseFloat(tok, 64)
			if err != nil {
				return nil, fmt.Errorf("invalid number at row %d col %d: %v ", i+1, j+1, err)
			}
			row[j] = val
		}
		matrix[i] = row
	}

	return matrix, nil
}
