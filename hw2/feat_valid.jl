using JSON
using LinearAlgebra
using Statistics
using Printf

include("readclassjson.jl")
makelist(x::Matrix) = Vector.(eachrow(x))

d = readclassjson("feat_valid.json")
utrain = makelist(d["utrain"])
utest  = makelist(d["utest"])
vtrain = d["vtrain"]
vtest  = d["vtest"]

function phi(u)
    u1, u2, u3 = u[1], u[2], u[3]
    return [u1, u2, u3, u1*u2, u2*u3, u1*u3, u1^2, u2^2, u3^2]
end

function softknn(x, U, v, rho)
    w = [exp(-sum((x .- U[i]).^2) / (2*rho^2)) for i in 1:length(U)]
    return sum(w .* v) / sum(w)
end

rmse(a, b) = sqrt(mean((a .- b).^2))

rhos = [0.25, 0.5, 1, 2]

for rho in rhos
    pred_train = [softknn(x, utrain, vtrain, rho) for x in utrain]
    pred_test = [softknn(x, utrain, vtrain, rho) for x in utest]
    error_train = rmse(pred_train, vtrain)
    error_test = rmse(pred_test, vtest)
    @printf("phi = u. rho: %.2f, RMSE Train: %.4f, RMSE Test: %.4f\n", rho, error_train, error_test)
end

for rho in rhos
    pred_train = [softknn(x, xtrain, vtrain, rho) for x in xtrain]
    pred_test = [softknn(x, xtrain, vtrain, rho) for x in xtest]
    error_train = rmse(pred_train, vtrain)
    error_test = rmse(pred_test, vtest)
    @printf("phi = (u1, u2, u3, u1*u2, u2*u3, u1*u3, u1^2, u2^2, u3^2). rho: %.2f, RMSE Train: %.4f, RMSE Test: %.4f\n", rho, error_train, error_test)
end

# feature mapping of phi = u with softknn rho = 0.25 gives the best performance on the test set with RMSE Test: 4.3814.
