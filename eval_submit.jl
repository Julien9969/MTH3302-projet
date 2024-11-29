using CSV, MLJ, DataFrames, Statistics


best_pred = CSV.read("current_best.csv", DataFrame, decimal='.')

current_pred = CSV.read("benchmark.csv", DataFrame, decimal='.')


println("rmse between best and current: ", rms(best_pred.consommation, current_pred.consommation))