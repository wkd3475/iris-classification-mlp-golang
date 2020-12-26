package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

func Relu(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return x
}

func ReluPrime(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return 1
}

func Softmax(m *mat.Dense) *mat.Dense {
	maxElement := mat.Max(m)
	subMaxElement := func(_, _ int, v float64) float64 {
		return v - maxElement
	}
	m.Apply(subMaxElement, m)
	numRows, numCols := m.Dims()

	expValue := mat.NewDense(numRows, numCols, nil)
	applyExp := func(_, _ int, v float64) float64 {
		return math.Exp(v)
	}
	expValue.Apply(applyExp, m)
	sumElement := mat.Sum(expValue)

	applySoftmax := func(_, _ int, v float64) float64 {
		return math.Exp(v) / sumElement
	}
	m.Apply(applySoftmax, m)

	return m
}

func CalLoss(output, y *mat.Dense) *mat.Dense {

	return nil
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

func (nn *neuralNet) train(x, y *mat.Dense) error {
	//randGen 초기화
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	//layer를 만들기위한 slice 구성
	wHiddenRaw := make([]float64, nn.config.hiddenNeurons*nn.config.inputNeurons)
	bHiddenRaw := make([]float64, nn.config.hiddenNeurons)
	wOutRaw := make([]float64, nn.config.outputNeurons*nn.config.hiddenNeurons)
	bOutRaw := make([]float64, nn.config.outputNeurons)

	//layer를 만들기위한 slice를 randGen을 사용해서 초기화
	for _, param := range [][]float64{wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}
	//layer 구성
	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, wHiddenRaw)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, bHiddenRaw)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, wOutRaw)
	bOut := mat.NewDense(1, nn.config.outputNeurons, bOutRaw)
	output := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)

	for i := 0; i < nn.config.numEpochs; i++ {

		//forward
		var hiddenLayerInput mat.Dense
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 {
			return v + bHidden.At(0, col)
		}
		hiddenLayerInput.Apply(addBHidden, &hiddenLayerInput)
		var hiddenLayerActivations mat.Dense
		applyRelu := func(_, _ int, v float64) float64 {
			return Relu(v)
		}
		hiddenLayerActivations.Apply(applyRelu, &hiddenLayerInput)

		var outputLayerInput mat.Dense
		outputLayerInput.Mul(&hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 {
			return v + bOut.At(0, col)
		}
		outputLayerInput.Apply(addBOut, &outputLayerInput)
		output = Softmax(&outputLayerInput)

		//get loss
		losses := CalLoss(output, y)
		loss := mat.Sum(losses)
		fmt.Printf("loss : %f", loss)
		// for j := 0; j < nn.config.outputNeurons; j++ {
		// 	l := -y.At(0, j) * math.Log(output.At(0, j))
		// 	losses[j] = l
		// 	loss += l
		// }
		// fmt.Print(loss)

		//backward
		// networkError := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, losses)

		// slopeOutputLayer := mat.NewDense(0, 0, nil)
		// applySigmoidPrime := func(_, _ int, v float64) float64 {
		// 	return SigmoidPrime(v)
		// }
		// slopeOutputLayer.Apply(applySigmoidPrime, output)
		// slopeHiddenLayer := mat.NewDense(0, 0, nil)
		// slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		// dOutput := mat.NewDense(0, 0, nil)
		// dOutput.MulElem(networkError, slopeOutputLayer)
		// errorAtHiddenLayer := mat.NewDense(0, 0, nil)
		// errorAtHiddenLayer.Mul(dOutput, wOut.T())

		// dHiddenLayer := mat.NewDense(0, 0, nil)
		// dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)
	}

	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

func main() {
	f, err := os.Open("train.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 5
	reader.LazyQuotes = true

	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	var inputsIndex int
	var labelsIndex int

	label2i := map[string]int{"setosa": 0, "Versicolor": 1, "virginica": 2}

	for idx, record := range rawCSVData {
		if idx == 0 {
			continue
		}

		for i, val := range record {
			if i == 4 {
				labelsMat := []float64{0.0, 0.0, 0.0}
				labelsMat[label2i[val]] = 1.0

				for j := 0; j < 3; j++ {
					labelsData[labelsIndex] = labelsMat[j]
					labelsIndex++
				}
				continue
			}

			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)

	config := neuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}

	// input := mat.NewDense(1, 4, []float64{
	// 	1.0, 0.0, 1.0, 0.0,
	// })

	// labels := mat.NewDense(3, 1, []float64{1.0, 1.0, 0.0})

	network := newNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	g := mat.Formatted(network.wHidden, mat.Prefix(" "))
	fmt.Printf("\nwHidden = %v\n\n", g)

	g = mat.Formatted(network.bHidden, mat.Prefix(" "))
	fmt.Printf("\nbHidden = %v\n\n", g)

	g = mat.Formatted(network.wOut, mat.Prefix(" "))
	fmt.Printf("\nwOut = %v\n\n", g)

	g = mat.Formatted(network.bOut, mat.Prefix(" "))
	fmt.Printf("\nbOut = %v\n\n", g)
}
