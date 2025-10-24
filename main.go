// Copyright 2025 The Eigenvector Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"fmt"
	"io"
	"math/cmplx"
	"sort"
	"strconv"

	"gonum.org/v1/gonum/mat"

	"github.com/pointlander/eigenvector/kmeans"
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Cluster  int
	Index    int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Inverse is the labels inverse map
var Inverse = [3]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

func main() {
	iris := Load()
	data := make([]float64, 0, 4*len(iris))
	for _, value := range iris {
		data = append(data, value.Measures...)
	}
	a := mat.NewDense(len(iris), 4, data)
	adj := mat.NewDense(len(iris), len(iris), nil)
	adj.Mul(a, a.T())
	var eig mat.Eigen
	ok := eig.Factorize(adj, mat.EigenRight)
	if !ok {
		fmt.Println("Eigenvalue decomposition failed.")
		return
	}
	eigenvectors := mat.NewCDense(len(iris), len(iris), nil)
	eig.VectorsTo(eigenvectors)
	data2 := make([]float64, 0, len(iris)*len(iris))
	vectors := make([][]float64, len(iris))
	for r := range len(iris) {
		row := make([]float64, 3)
		fmt.Println(eigenvectors.At(r, 0))
		for c := range len(iris) {
			data2 = append(data2, cmplx.Abs(eigenvectors.At(r, c)))
		}
		for c := range 3 {
			row[c] = cmplx.Abs(eigenvectors.At(r, c))
		}
		vectors[r] = row
	}
	b := mat.NewDense(len(iris), len(iris), data2)
	_ = b

	meta := make([][]float64, len(iris))
	for i := range meta {
		meta[i] = make([]float64, len(iris))
	}
	const k = 3
	for i := 0; i < 33; i++ {
		clusters, _, err := kmeans.Kmeans(int64(i+1), vectors, k, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := 0; i < len(meta); i++ {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					meta[i][j]++
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range clusters {
		iris[i].Cluster = clusters[i]
	}
	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Cluster < iris[j].Cluster
	})
	for i := range iris {
		fmt.Println(iris[i].Cluster, iris[i].Label)
	}
}
