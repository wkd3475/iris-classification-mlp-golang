package main

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestRelu(t *testing.T) {
	a := -4.56
	result := Relu(a)
	if result != 0.0 {
		t.Errorf("expected:%f actual:%f", 0.0, result)
	}
}

func TestSoftmax(t *testing.T) {
	a := mat.NewDense(1, 4, []float64{
		1.0, 1.0, 1.0, 1.0,
	})
	b := mat.NewDense(1, 4, []float64{
		0.25, 0.25, 0.25, 0.25,
	})
	result := mat.Equal(Softmax(a), b)
	if !result {
		t.Errorf("expected:%t actual:%t", true, false)
	}
}
