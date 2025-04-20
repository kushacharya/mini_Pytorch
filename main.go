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
		log.Fatalf("Error getting tensot B : %v", err)
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

	result, err := a.MatMul(b)
	if err != nil {
		log.Fatalf("error in matrix multiplication : %v", err)
	}

	fmt.Println("Result of the matrix multiplication : ")
	result.Print()
}
