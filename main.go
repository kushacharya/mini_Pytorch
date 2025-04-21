package main

import (
	"Github_campaign/Loader"
	"Github_campaign/Tensor"
	"fmt"
	"log"
)

func main() {
	// I have created a simple Tensor 2D slice here
	aData, err := Loader.GetUserMatrix("A")
	if err != nil {
		log.Fatalf("Error getting ternsor A : %v", err)
	}
	bData, err := Loader.GetUserMatrix("B")
	if err != nil {
		log.Fatalf("Error getting tensor B : %v", err)
	}
	// here it is returning two things: 1. a and 2. err
	a, err := Tensor.NewTensor(aData)
	// I am checking here that whether the error occurred or not
	if err != nil {
		fmt.Println("Error creating Tensor A", err)
		return
	}

	b, err := Tensor.NewTensor(bData)
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
}
