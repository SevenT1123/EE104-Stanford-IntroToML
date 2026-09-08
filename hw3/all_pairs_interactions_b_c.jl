include("readclassjson.jl")
using LinearAlgebra
using Statistics

data = readclassjson("all_pairs_data.json")
U = data["U"]
v = data["v"]

n = size(U, 1)
u1, u2, u3 = U[:,1], U[:,2], U[:,3]

phi = hcat(ones(n), u1, u2, u3, u1 .* u2, u1 .* u3, u2 .* u3)

ntrain = div(n, 2)
phi_train, phi_test = phi[1:ntrain,:], phi[ntrain+1:end,:]
vtrain, vtest = v[1:ntrain], v[ntrain+1:end]

theta_b = phi_train \ vtrain
println("theta_b = ", theta_b)

rmse(phi, v, theta_b) = sqrt(mean((phi * theta_b - v).^2))

train_rmse_b = rmse(phi_train, vtrain, theta_b)
test_rmse_b = rmse(phi_test, vtest, theta_b)

println("train RMSE = ", train_rmse_b)
println("test RMSE = ", test_rmse_b)

# Remove u2 and u3
phi_reduced = [phi[:,1:2] phi[:,5:end]]
phi_reduced_train, phi_reduced_test = phi_reduced[1:ntrain,:], phi_reduced[ntrain+1:end,:]

theta_c = phi_reduced_train \ vtrain
println("theta_c = ", theta_c)

train_rmse_c = rmse(phi_reduced_train, vtrain, theta_c)
test_rmse_c = rmse(phi_reduced_test, vtest, theta_c)

println("train RMSE = ", train_rmse_c)
println("test RMSE = ", test_rmse_c)
