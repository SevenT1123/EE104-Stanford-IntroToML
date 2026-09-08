import Pkg
Pkg.add(["Plots", "JSON"])
using Plots

include("readclassjson.jl")
d = readclassjson("sine.json")
x = d["x"]
y = d["y"]

function knn(X, Y, x, k)
    n = size(X)[1]
    m = size(x)[1]
    y_hat = zeros(m)
    for j = 1:m
        dists_sq = [(X[i] .- x[j]).^2 for i = 1:n]
        nearest_neighbors_idxs = sortperm(dists_sq)[1:k]
        y_hat[j] = sum(Y[nearest_neighbors_idxs]) / k
    end
    return y_hat
end

function softknn(X, Y, x, rho)
    n = size(X)[1]
    m = size(x)[1]
    y_hat = zeros(m)
    for j = 1:m
        exp_weights = [exp(-((X[i] .- x[j]).^2) / rho) for i = 1:n]
        w = exp_weights / sum(exp_weights)
        y_hat[j] = sum(w .* Y)
    end
    return y_hat
end

k = [1, 2, 3]
rho = [0.01, 0.02, 0.03, 0.05, 0.1]
N = 500

sample_points = collect(range(0, 1; length=N))
x_true = range(minimum(x), maximum(x), length=200)
y_true = sin.(10 .* x_true)
y_true_test = sin.(10 .* sample_points)

function rms_error(y_hat, y_true)
    N = length(y_true)
    return sqrt(sum((y_hat .- y_true).^2) / N)
end

for i in k
    p = plot(x_true, y_true, label="f(x) = sin(10x)", xlabel="x", ylabel="y", title="Sine Plot (k=$i)", linewidth=2)
    y_hat = knn(x, y, x, i)
    scatter!(p, x, y, label="Data points", marker=:circle, ms=3, color=:black)
    scatter!(p, x, y_hat, label="k=$i", linewidth=2)
    display(p)
end

for i in rho
    p = plot(x_true, y_true, label="f(x) = sin(10x)", xlabel="x", ylabel="y", title="Sine Plot (rho=$i)", linewidth=2)
    y_hat = softknn(x, y, x, i)
    scatter!(p, x, y, label="Data points", marker=:circle, ms=3, color=:black)
    scatter!(p, x, y_hat, label="rho=$i", linewidth=2)
    display(p)
end

for i in k 
    y_hat = knn(x, y, sample_points, i)
    err = rms_error(y_hat, y_true_test)
    println("RMS error for k=$i: $err")
end 

for i in rho 
    y_hat = softknn(x, y, sample_points, i)
    err = rms_error(y_hat, y_true_test)
    println("RMS error for rho=$i: $err")
end