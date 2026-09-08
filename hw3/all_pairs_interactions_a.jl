include("readclassjson.jl")
using LinearAlgebra
using Statistics

data = readclassjson("all_pairs_data.json")
U = data["U"]
v = data["v"]

n = size(U, 1)

Ufull = hcat(ones(n), U)

ntrain = div(n, 2)
Utrain, Utest = Ufull[1:ntrain,:], Ufull[ntrain+1:end,:]
vtrain, vtest = v[1:ntrain], v[ntrain+1:end]

theta = Utrain \ vtrain
println("theta = ", theta)

rsme(U, v, theta) = sqrt(mean((U * theta - v).^2))

train_rmse = rsme(Utrain, vtrain, theta)
test_rmse = rsme(Utest, vtest, theta)

println("train RMSE = ", train_rmse)
println("test RMSE = ", test_rmse)