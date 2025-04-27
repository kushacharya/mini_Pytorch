package main

import (
	"fmt"
	"log"
	"mini_pytorch/loader"
	"mini_pytorch/tensor"
)

func main() {
	// I have created a simple Tensor 2D slice here
	aData, err := loader.GetUserMatrix("A")
	if err != nil {
		log.Fatalf("Error getting ternsor A : %v", err)
	}
	bData, err := loader.GetUserMatrix("B")
	if err != nil {
		log.Fatalf("Error getting tensor B : %v", err)
	}
	// here it is returning two things: 1. a and 2. err
	a, err := tensor.NewTensor(aData)
	// I am checking here that whether the error occurred or not
	if err != nil {
		fmt.Println("Error creating Tensor A", err)
		return
	}

	b, err := tensor.NewTensor(bData)
	if err != nil {
		fmt.Println("Error creating Tensor B", err)
		return
	}

	// Print is defined in the Tensor.go as it is returning the whole matrix in for loop.
	fmt.Println("Tensor A:")
	a.Print()
	// same here I am calling the print func in Tensor.go
	fmt.Println("Tensor B:")
	b.Print()

	matmulCheck, err := a.MatMul(b)
	if err != nil {
		log.Fatalf("error in matrix multiplication : %v", err)
	}

	fmt.Println("Result of the matrix multiplication : ")
	matmulCheck.Print()

	add, err := a.Add(b)
	if err != nil {
		log.Fatalf("error in addition of matrix : %v", err)
	}

	fmt.Println("Result of the matrix addition : ")
	add.Print()

	sub, err := a.Sub(b)
	if err != nil {
		log.Fatalf("error in subtraction of matrix : %v", err)
	}

	fmt.Println("Result of the matrix subtraction : ")
	sub.Print()

	mul, err := a.Multiply(b)
	if err != nil {
		log.Fatalf("error in multiply (Element wise) of matrix : %v", err)
	}

	fmt.Println("Result of the matrix multiplication : ")
	mul.Print()

	// TODO: Hardcoded value
	addS, err := a.AddScalar(5.0)
	if err != nil {
		log.Fatalf("error in the AddScalar function: %v", err)
	}
	fmt.Println("Result after adding the scalar value")
	addS.Print()

	trans, err := a.Transpose()
	if err != nil {
		log.Fatalf("error in tansposing the matrix: %v", err)
	}
	fmt.Println("Result of transpose matrix")
	trans.Print()

	rows, cols, err := a.Shape()
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	fmt.Printf("Shape of matrix  Rows :%d , Cols: %d", rows, cols)

	random, err := tensor.Random(4, 4)
	if err != nil {
		log.Fatalf("error in generatting the random matrix %v", err)
	}
	random.Print()
}
