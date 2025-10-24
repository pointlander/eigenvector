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
	"math"
	"math/cmplx"
	"math/rand"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"

	"github.com/pointlander/eigenvector/kmeans"
	"github.com/pointlander/gradient/tf64"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-1
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
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
		row := make([]float64, 2)
		fmt.Println(eigenvectors.At(r, 0))
		for c := range len(iris) {
			data2 = append(data2, cmplx.Abs(eigenvectors.At(r, c)))
		}
		for c := range 2 {
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

	{
		rng := rand.New(rand.NewSource(1))
		iris := Load()
		set := tf64.NewSet()
		set.Add("a", 4, len(iris))
		a := set.ByName["a"]
		for _, row := range iris {
			a.X = append(a.X, row.Measures...)
		}

		for ii := range set.Weights {
			w := set.Weights[ii]
			if strings.HasPrefix(w.N, "a") {
				w.States = make([][]float64, StateTotal)
				for ii := range w.States {
					w.States[ii] = make([]float64, len(w.X))
				}
				continue
			}
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:cap(w.X)]
				w.States = make([][]float64, StateTotal)
				for ii := range w.States {
					w.States[ii] = make([]float64, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for range cap(w.X) {
				w.X = append(w.X, rng.NormFloat64()*factor)
			}
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
		}

		drop := .3
		dropout := map[string]interface{}{
			"rng":  rng,
			"drop": &drop,
		}

		sa := tf64.T(tf64.Mul(tf64.Dropout(tf64.Mul(set.Get("a"), set.Get("a")), dropout), tf64.T(set.Get("a"))))
		loss := tf64.Avg(tf64.Quadratic(set.Get("a"), sa))

		for iteration := range 256 {
			pow := func(x float64) float64 {
				y := math.Pow(x, float64(iteration+1))
				if math.IsNaN(y) || math.IsInf(y, 0) {
					return 0
				}
				return y
			}

			set.Zero()
			l := tf64.Gradient(loss).X[0]
			if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
				fmt.Println(iteration, l)
				return
			}

			norm := 0.0
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = math.Sqrt(norm)
			b1, b2 := pow(B1), pow(B2)
			scaling := 1.0
			if norm > 1 {
				scaling = 1 / norm
			}
			for _, w := range set.Weights {
				for ii, d := range w.D {
					g := d * scaling
					m := B1*w.States[StateM][ii] + (1-B1)*g
					v := B2*w.States[StateV][ii] + (1-B2)*g*g
					w.States[StateM][ii] = m
					w.States[StateV][ii] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
			fmt.Println(l)
		}

		vectors := make([][]float64, len(iris))
		for i := range vectors {
			row := make([]float64, 4)
			for ii := range row {
				row[ii] = a.X[i*4+ii]
			}
			vectors[i] = row
		}

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
}
