package main

import (
	"bytes"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type Result struct {
	Name        string
	Probability float64
}

func main() {
	http.HandleFunc("/mnist", mnistHandler)
	http.ListenAndServe(":3000", nil)
}

func Recognition(tensor *tensorflow.Tensor) (string, error) {
	var probability float64
	// tf.saved_model.builder in Pythonで構築したモデルを呼び出す
	model, err := tensorflow.LoadSavedModel("mymnist", []string{"mnisttag"}, nil)
	if err != nil {
		return "", err
	}
	defer model.Session.Close()

	// inputするimageのshapeをtensorに変換する [batch size][width][height][channels]
	result, err := model.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			model.Graph.Operation("conv2d_1_input").Output(0): tensor,
		},
		[]tensorflow.Output{
			model.Graph.Operation("dense_2/Softmax").Output(0),
		},
		nil,
	)

	if err != nil {
		return "", err
	}
	labels := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
	probabilities := result[0].Value().([][]float32)[0]
	max := 0
	for i, v := range probabilities {
		if float64(v) > probability {
			probability = float64(probabilities[i])
			max = i
		}
	}
	log.Println(probability, max)
	log.Println(probabilities)
	return labels[max], nil
}

func mnistHandler(w http.ResponseWriter, r *http.Request) {
	imageFile, header, err := r.FormFile("image")
	if err != nil {
		log.Println(err)
		return
	}
	defer imageFile.Close()

	imageName := strings.Split(header.Filename, ".")
	var imageBuffer bytes.Buffer
	io.Copy(&imageBuffer, imageFile)
	tensor, err := ConvertImageToTensor(&imageBuffer, imageName[:1][0])
	if err != nil {
		log.Println(err)
		return
	}
	probability, err := Recognition(tensor)
	if err != nil {
		log.Println(err)
	}
	log.Println(probability)
}

func ConvertImageToTensor(imageBuffer *bytes.Buffer, format string) (*tensorflow.Tensor, error) {
	tensor, err := tensorflow.NewTensor(imageBuffer.String())
	if err != nil {
		return nil, err
	}
	graph, input, output, err := makeTransFormImageGraph(format)
	if err != nil {
		return nil, err
	}
	session, err := tensorflow.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{input: tensor},
		[]tensorflow.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func makeTransFormImageGraph(format string) (graph *tensorflow.Graph, input, output tensorflow.Output, err error) {
	const (
		Height, Width = 28, 28
		Batch         = float32(1)
		Scale         = float32(1)
	)
	s := op.NewScope()
	input = op.Placeholder(s, tensorflow.String)
	var decode tensorflow.Output
	decode = op.DecodeJpeg(s, input, op.DecodeJpegChannels(1)) //0,1だけなので1

	output = op.Div(s,
		op.Sub(s,
			// Resize to 224x224 with bilinear interpolation
			op.ResizeBilinear(s,
				// Create a batch containing a single image
				op.ExpandDims(s,
					// Use decoded pixel values
					op.Cast(s, decode, tensorflow.Float),
					op.Const(s.SubScope("make_batch"), int32(1))),
				op.Const(s.SubScope("size"), []int32{Height, Width})),
			op.Const(s.SubScope("mean"), Batch)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()
	return graph, input, output, err
}
