include("readclassjson.jl")
using LinearAlgebra
using Statistics
using Random
using Plots

data = readclassjson("wildfire_data.json")
U = data["U"]
v = data["v"]

n, d = size(U)
Random.seed!(1234)
perm = randperm(n)
# randomly partitoned 80% of data for training
ntrain = round(Int, 0.8 * n) 
train_idx = perm[1:ntrain]
test_idx = perm[ntrain+1:end]

# Using training mean and std for standardization of test set
function standardize_fit_transform(Xtrain, Xtest)
    Xtrain_mean = mean(Xtrain, dims=1)
    Xtrain_std = std(Xtrain, dims=1)
    zero_cols = findall(vec(Xtrain_std) .== 0)
    if !isempty(zero_cols)
        println("Warning: training columns ", zero_cols, " are constant (Xtrain_std=0); leaving them un-scaled.")
        Xtrain_std[1, zero_cols] .= 1.0
    end
    Xtrain_standardized = (Xtrain .- Xtrain_mean) ./ Xtrain_std
    Xtest_standardized  = (Xtest  .- Xtrain_mean) ./ Xtrain_std
    return Xtrain_standardized, Xtest_standardized, Xtrain_mean, Xtrain_std
end

# columns of U are: [posX, posY, month, FFMC, temp, wind, rain]
Utrain, Utest = U[train_idx,:], U[test_idx,:]
vtrain, vtest = v[train_idx], v[test_idx]

Utrain_standardized, Utest_standardized, Utrain_mean, Utrain_std = standardize_fit_transform(Utrain, Utest)
Utrain_standardized_mean, Utrain_standardized_std = mean(Utrain_standardized, dims=1), std(Utrain_standardized, dims=1)

MONTH_COL = 3
OTHER_COLS = [1, 2, 4, 5, 6, 7]

function sinusoidal_embeddings(x) 
    n = size(x, 1)
    S = zeros(n, 2)
    S[:, 1] = sin.(2 * pi .* x ./ 12) 
    S[:, 2] = cos.(2 * pi .* x ./ 12)
    return S
end

# One-hot encode the month column and standardize the other columns
Strain = sinusoidal_embeddings(Utrain[:, MONTH_COL])
Stest = sinusoidal_embeddings(Utest[:, MONTH_COL])
Strain_standardized, Stest_standardized, Strain_mean, Strain_std = standardize_fit_transform(Strain, Stest)

Otrain = Utrain[:, OTHER_COLS]
Otest = Utest[:, OTHER_COLS]
Otrain_standardized, Otest_standardized, Otrain_mean, Otrain_std = standardize_fit_transform(Otrain, Otest)

ones_train = ones(size(train_idx, 1), 1)
ones_test = ones(size(test_idx, 1), 1)

# Assemble the final training and test matrices by concatenating the standardized month embeddings, standardized other features, and a column of ones for the bias term
Xtrain = hcat(Strain_standardized, Otrain_standardized, ones_train)
Xtest = hcat(Stest_standardized, Otest_standardized, ones_test)

feature_names = vcat(["month_$(i)" for i in 1:12], ["posX", "posY", "FFMC", "temp", "wind", "rain"], ["intercept"])

println("\nXtrain size: ", size(Xtrain), " Xtest size: ", size(Xtest))
println("Number of features: ", size(Xtrain, 2))
println("\nFirst row of Xtrain (sinusoidal embedded):")
for(name, val) in zip(feature_names, Xtrain[1, :])
    println("Feature: ", name, " Value: ", val)
end

lambdas = 10 .^range(-1, stop=5, length=100)

# Ridge regression with one-hot-encoded month features
n_features = size(Xtrain, 2)
bias_idx = n_features

XtX = transpose(Xtrain) * Xtrain
Xty = transpose(Xtrain) * vtrain

train_rmse = zeros(length(lambdas))
test_rmse = zeros(length(lambdas))

for(i, lambda) in enumerate(lambdas)
    D = lambda * Matrix{Float64}(I, n_features, n_features)
    D[bias_idx, bias_idx] = 0.0
    theta = (XtX + D) \ Xty
    train_pred = Xtrain * theta
    test_pred = Xtest * theta
    train_rmse[i] = sqrt(mean((train_pred - vtrain).^2))
    test_rmse[i] = sqrt(mean((test_pred - vtest).^2))
end

best_train_idx = argmin(train_rmse)
best_test_idx = argmin(test_rmse)

println("\nMin train RMSE: ", train_rmse[best_train_idx], " at lambda = ", lambdas[best_train_idx])
println("Min test RMSE: ", test_rmse[best_test_idx], " at lambda = ", lambdas[best_test_idx])

plt = plot(lambdas, train_rmse, xscale=:log10, label="Train RMSE", lw=2, xlabel="Lambda", ylabel="RMSE", title="Ridge Regression RMSE vs Lambda", legend=:topleft)
plot!(plt, lambdas, test_rmse, label="Test RMSE", lw=2)
scatter!(plt, [lambdas[best_train_idx]], [train_rmse[best_train_idx]], label="Best Train RMSE", ms=6, color=:green)
scatter!(plt, [lambdas[best_test_idx]], [test_rmse[best_test_idx]], label="Best Test RMSE", ms=6, color=:yellow)