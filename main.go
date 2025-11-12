// Copyright 2025 The Eigenvector Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"flag"
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

// Random generates a random iris data set
func Random(seed int64) []Fisher {
	fisher, rng := make([]Fisher, 150), rand.New(rand.NewSource(seed))
	for i := range fisher {
		fisher[i].Measures = make([]float64, 4)
		for ii := range fisher[i].Measures {
			fisher[i].Measures[ii] = rng.Float64()
		}
		fisher[i].Label = fmt.Sprintf("%d", i)
		fisher[i].Index = i
	}
	return fisher
}

var (
	// FlagAutoEncoder is the autoencoder mode
	FlagAutoEncoder = flag.Bool("ae", false, "autoencoder mode")
	// FlagInverseSelfAttention is the inverse self attention mode
	FlagInverseSelfAttention = flag.Bool("isa", false, "inverse self attention mode")
	// FlagMPR is the markov page rank mode
	FlagMPR = flag.Bool("mpr", false, "markov page rank mode")
	// FlagM1PR is the order 1 markov page rank mode
	FlagM1PR = flag.Bool("m1pr", false, "order 1 markov page rank mode")
)

// AutoEncoderMode is the autoencoder mode
func AutoEncoderMode() {
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

// InverseSelfAttentionMode is the inverse self attention mode
func InverseSelfAttentionMode() {
	const (
		Eta = 1.0e-3
	)
	rng := rand.New(rand.NewSource(1))
	iris := Load()
	others := tf64.NewSet()
	others.Add("x", 4, len(iris))
	x := others.ByName["x"]
	for _, row := range iris {
		x.X = append(x.X, row.Measures...)
	}

	set := tf64.NewSet()
	set.Add("i", 4, len(iris))
	set.Add("j", 4, len(iris))

	for ii := range set.Weights {
		w := set.Weights[ii]
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
			w.X = append(w.X, rng.NormFloat64()*factor*.01)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	sa := tf64.T(tf64.Mul(tf64.Mul(set.Get("i"), set.Get("j")), tf64.T(others.Get("x"))))
	loss := tf64.Avg(tf64.Quadratic(others.Get("x"), sa))

	for iteration := range 2 * 1024 {
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

	meta := make([][]float64, len(iris))
	for i := range meta {
		meta[i] = make([]float64, len(iris))
	}
	const k = 3

	{
		y := set.ByName["i"]
		vectors := make([][]float64, len(iris))
		for i := range vectors {
			row := make([]float64, 4)
			for ii := range row {
				row[ii] = y.X[i*4+ii]
			}
			vectors[i] = row
		}
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
	}
	{
		y := set.ByName["j"]
		vectors := make([][]float64, len(iris))
		for i := range vectors {
			row := make([]float64, 4)
			for ii := range row {
				row[ii] = y.X[i*4+ii]
			}
			vectors[i] = row
		}
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
	acc := make(map[string][3]int)
	for i := range iris {
		fmt.Println(iris[i].Cluster, iris[i].Label)
		counts := acc[iris[i].Label]
		counts[iris[i].Cluster]++
		acc[iris[i].Label] = counts
	}
	for i, v := range acc {
		fmt.Println(i, v)
	}
}

// MPRMode is the markov page rank mode
func MPRMode() {
	dot := func(a, b []float64) float64 {
		x := 0.0
		for i, value := range a {
			x += value * b[i]
		}
		return x
	}

	rng := rand.New(rand.NewSource(1))
	iris := Load()
	length := len(iris)
	/*for i := range iris {
		max := 0.0
		for _, value := range iris[i].Measures {
			if value > max {
				max = value
			}
		}
		for ii, value := range iris[i].Measures {
			iris[i].Measures[ii] = value / max
		}
	}*/
	adj := make([]float64, length*length)
	for i := range length {
		for ii := range length {
			adj[i*length+ii] = dot(iris[i].Measures, iris[ii].Measures)
		}
	}
	for i := range length {
		sum := 0.0
		a := adj[i*length : (i+1)*length]
		for _, value := range a {
			sum += value
		}
		for ii := range a {
			a[ii] /= sum
		}
	}
	markov := make([]float64, length*length)
	node := 0
	for range 8 * 1024 * 1024 {
		total, selected := 0.0, rng.Float64()
		a := adj[node*length : (node+1)*length]
		for i, value := range a {
			total += value
			if selected < total {
				markov[i*length+node]++
				node = i
				break
			}
		}
	}

	type Point struct {
		Value float64
		Index int
	}
	reduction := func(points []Point) int {
		sort.Slice(points, func(i, j int) bool {
			return points[i].Value < points[j].Value
		})
		varab := 0.0
		avgab := 0.0
		for _, v := range points {
			avgab += v.Value
		}
		avgab /= float64(len(points))
		for _, v := range points {
			diff := avgab - v.Value
			varab += diff * diff
		}
		varab /= float64(len(points))
		max, index := 0.0, 0
		for i := 1; i < len(points)-1; i++ {
			vara, varb := 0.0, 0.0
			avga, avgb := 0.0, 0.0
			ca, cb := 0.0, 0.0
			for _, value := range points[:i] {
				avga += value.Value
				ca++
			}
			avga /= ca
			for _, value := range points[i:] {
				avgb += value.Value
				cb++
			}
			avgb /= cb
			for _, value := range points[:i] {
				diff := avga - value.Value
				vara += diff * diff
			}
			vara /= ca
			for _, value := range points[i:] {
				diff := avgb - value.Value
				varb += diff * diff
			}
			varb /= cb
			if diff := varab - (vara + varb); diff > max {
				max, index = diff, i
			}
		}
		return index
	}
	points := make([][]Point, length)
	data := make([][]float64, length)
	for i := range data {
		data[i] = markov[i*length : (i+1)*length]
		for ii := range data[i] {
			points[ii] = append(points[ii], Point{
				Value: data[i][ii],
				Index: i,
			})
		}
	}
	avg := make([]float64, length)
	for _, v := range data {
		for ii, value := range v {
			avg[ii] += value
		}
	}
	for i := range avg {
		avg[i] /= float64(len(data))
	}
	stddev := make([]float64, length)
	for _, v := range data {
		for ii, value := range v {
			diff := value - avg[ii]
			stddev[ii] += diff * diff
		}
	}
	for i := range stddev {
		stddev[i] = math.Sqrt(stddev[i] / float64(len(data)))
	}
	type Column struct {
		Index  int
		Stddev float64
	}
	columns := make([]Column, length)
	for i := range columns {
		columns[i].Index = i
		columns[i].Stddev = stddev[i]
	}
	sort.Slice(columns, func(i, j int) bool {
		return columns[i].Stddev > columns[j].Stddev
	})
	s1 := reduction(points[columns[0].Index])
	s2 := reduction(points[columns[0].Index][:s1])
	s3 := reduction(points[columns[0].Index][s1:])
	data2 := make([][]float64, length)
	for i := range data2 {
		for ii := range columns[:3] {
			data2[i] = append(data2[i], markov[i*length+columns[ii].Index])
		}
	}
	meta := make([][]float64, len(iris))
	for i := range meta {
		meta[i] = make([]float64, len(iris))
	}
	const k = 3
	for i := 0; i < 33; i++ {
		clusters, _, err := kmeans.Kmeans(int64(i+1), data2, k, kmeans.SquaredEuclideanDistance, -1)
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
	for i, value := range clusters {
		iris[i].Cluster = value
	}
	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Cluster < iris[j].Cluster
	})
	acc := make(map[string][3]int)
	for i := range iris {
		fmt.Println(iris[i].Cluster, iris[i].Label)
		counts := acc[iris[i].Label]
		counts[iris[i].Cluster]++
		acc[iris[i].Label] = counts
	}
	for i, v := range acc {
		fmt.Println(i, v)
	}
	fmt.Println(s1, s2, s3)
	for i := range s1 {
		fmt.Println(0, iris[points[columns[0].Index][i].Index].Label)
	}
	for i := s1; i < s1+s3; i++ {
		fmt.Println(1, iris[points[columns[0].Index][i].Index].Label)
	}
	for i := s1 + s3; i < len(iris); i++ {
		fmt.Println(2, iris[points[columns[0].Index][i].Index].Label)
	}
}

// M1PR is the order 1 markov page rank model
func M1PRMode() {
	dot := func(a, b []float64) float64 {
		x := 0.0
		for i, value := range a {
			x += value * b[i]
		}
		return x
	}

	rng := rand.New(rand.NewSource(1))
	iris := Load()
	length := len(iris)
	/*for i := range iris {
		max := 0.0
		for _, value := range iris[i].Measures {
			if value > max {
				max = value
			}
		}
		for ii, value := range iris[i].Measures {
			iris[i].Measures[ii] = value / max
		}
	}*/
	type Node struct {
		Weights []float64
		Markov  [][]float64
	}
	nodes := make([]Node, length)
	for i := range nodes {
		nodes[i].Weights = make([]float64, length)
		for ii := range nodes[i].Weights {
			nodes[i].Weights[ii] = dot(iris[i].Measures, iris[ii].Measures)
		}
		sum := 0.0
		for _, value := range nodes[i].Weights {
			sum += value
		}
		for ii := range nodes[i].Weights {
			nodes[i].Weights[ii] /= sum
		}
		nodes[i].Markov = make([][]float64, length)
		for ii := range nodes[i].Markov {
			nodes[i].Markov[ii] = make([]float64, length)
		}
	}

	node, previous := 0, 0
	for range 512 * 1024 * 1024 {
		total, selected := 0.0, rng.Float64()
		for i, value := range nodes[node].Weights {
			total += value
			if selected < total {
				nodes[i].Markov[node][previous]++
				node, previous = i, node
				break
			}
		}
	}

	meta := make([][]float64, len(iris))
	for i := range meta {
		meta[i] = make([]float64, len(iris))
	}
	const k = 3
	for i := range nodes {
		clusters, _, err := kmeans.Kmeans(int64(i+1), nodes[i].Markov, k, kmeans.SquaredEuclideanDistance, -1)
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
	for i, value := range clusters {
		iris[i].Cluster = value
	}
	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Cluster < iris[j].Cluster
	})
	acc := make(map[string][3]int)
	for i := range iris {
		fmt.Println(iris[i].Cluster, iris[i].Label)
		counts := acc[iris[i].Label]
		counts[iris[i].Cluster]++
		acc[iris[i].Label] = counts
	}
	for i, v := range acc {
		fmt.Println(i, v)
	}
}

func main() {
	flag.Parse()

	if *FlagAutoEncoder {
		AutoEncoderMode()
		return
	}

	if *FlagInverseSelfAttention {
		InverseSelfAttentionMode()
		return
	}

	if *FlagMPR {
		MPRMode()
		return
	}

	if *FlagM1PR {
		M1PRMode()
		return
	}

	const (
		// S is the scaling factor for the softmax
		S = 1.0 - 1e-300
	)

	softmax := func(values []float64) {
		max := 0.0
		for _, v := range values {
			if v > max {
				max = v
			}
		}
		s := max * S
		sum := 0.0
		for j, value := range values {
			values[j] = math.Exp(value - s)
			sum += values[j]
		}
		for j, value := range values {
			values[j] = value / sum
		}
	}

	dot := func(a, b []float64) float64 {
		x := 0.0
		for i, value := range a {
			x += value * b[i]
		}
		return x
	}

	cs := func(a, b []float64) float64 {
		ab := dot(a, b)
		aa := dot(a, a)
		bb := dot(b, b)
		if aa <= 0 {
			return 0
		}
		if bb <= 0 {
			return 0
		}
		return ab / (math.Sqrt(aa) * math.Sqrt(bb))
	}

	if *FlagMPR {

		return
	}

	process := func(iris []Fisher, cluster bool) float64 {
		data := make([]float64, 0, 4*len(iris))
		for _, value := range iris {
			data = append(data, value.Measures...)
		}
		a := mat.NewDense(len(iris), 4, data)
		adj := mat.NewDense(len(iris), len(iris), nil)
		adj.Mul(a, a.T())
		cp := mat.NewDense(len(iris), len(iris), nil)
		cp.Copy(adj)
		for r := range len(iris) {
			row := make([]float64, len(iris))
			for ii := range row {
				row[ii] = cp.At(r, ii)
			}
			softmax(row)
			cp.SetRow(r, row)
		}
		x := mat.NewDense(len(iris), 4, nil)
		x.Mul(cp, a)
		var eig mat.Eigen
		ok := eig.Factorize(adj, mat.EigenRight)
		if !ok {
			fmt.Println("Eigenvalue decomposition failed.")
			return 0
		}
		eigenvectors := mat.NewCDense(len(iris), len(iris), nil)
		eig.VectorsTo(eigenvectors)
		data2 := make([]float64, 0, len(iris)*len(iris))
		vectors := make([][]float64, len(iris))
		i, j := make([]float64, 0, len(iris)), make([]float64, 0, len(iris))
		for r := range len(iris) {
			row := make([]float64, 2)
			fmt.Println(eigenvectors.At(r, 0))
			i = append(i, cmplx.Abs(eigenvectors.At(r, 0)))
			j = append(j, x.At(r, 0))
			for c := range len(iris) {
				data2 = append(data2, cmplx.Abs(eigenvectors.At(r, c)))
			}
			for c := range 2 {
				row[c] = cmplx.Abs(eigenvectors.At(r, c))
			}
			vectors[r] = row
		}

		if !cluster {
			return cs(i, j)
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
		return cs(i, j)
	}

	iris := Load()
	fmt.Println(process(iris, true))
	results := make([]float64, 128)
	count := 0
	for i := range results {
		iris := Random(int64(i + 1))
		cs := process(iris, false)
		results[i] = cs
		if cs < .95 {
			count++
		}
	}
	for _, value := range results {
		fmt.Println(value)
	}
	fmt.Println(count)
}
