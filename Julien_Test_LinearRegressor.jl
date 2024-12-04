using CSV
using DataFrames
using MLJ
using Random
using Statistics
using MLJDecisionTreeInterface

Random.seed!(4355)

train_data = CSV.read("train.csv", DataFrame, decimal=',')
test_data = CSV.read("test.csv", DataFrame, decimal=',')

# consommation au carre
# train_data[!,:consommation] = train_data[:, :consommation] .^ 2


function remove_outliers(data, column, alpha=1.5)
    unique_values = unique(data[:, column])
    for value in unique_values
        subset = data[data[:, column] .== value, :]
        if size(subset, 1) == 0
            continue
        end
        q1 = quantile(subset[:, :consommation], 0.25)
        q3 = quantile(subset[:, :consommation], 0.75)
        iqr = q3 - q1
        lower_bound = q1 - alpha * iqr
        upper_bound = q3 + alpha * iqr
        data = data[(data[:, column] .!= value) .| ((data[:, column] .== value) .& (data[:, :consommation] .>= lower_bound) .& (data[:, :consommation] .<= upper_bound)), :]
    end
    return data
end

filtered_train = copy(train_data)

# best=0.4 0.5641931723004245

println("Before removing outliers: ", size(train_data, 1))

alpha = 1.5
filtered_train = remove_outliers(filtered_train, :annee, alpha);
filtered_train = remove_outliers(filtered_train, :type, alpha);
filtered_train = remove_outliers(filtered_train, :nombre_cylindres, alpha);
filtered_train = remove_outliers(filtered_train, :transmission, alpha);
filtered_train = remove_outliers(filtered_train, :boite, alpha);
println("After removing outliers: ", size(filtered_train, 1))


schema(filtered_train)

# describe(filtered_train[:, :consommation])


# remove annee
# filtered_train = filtered_train[:, Not(:boite)]
# test_data = test_data[:, Not(:boite)]

# filtered_train[!, :annee] = Float64.(filtered_train[:, :annee])
categorical_cols = [:type, :transmission, :boite, :annee]

function one_hot_encode(data, columns)
    for column in columns
        if column in ["consommation", "cylindree"]
            continue
        end
        coerce!(data, column => Multiclass)
    end

    one_hot = machine(OneHotEncoder(), data)
    fit!(one_hot, verbosity=0)
    data = MLJ.transform(one_hot, data)
    return data
end

filtered_train = one_hot_encode(filtered_train, categorical_cols)
filtered_train[!,:nombre_cylindres] = Float64.(filtered_train[:, :nombre_cylindres])

# filtered_train[!,:annee] = Float64.(filtered_train[:, :annee])
# println(schema(filtered_train))
# println(eltype(filtered_train[:, :nombre_cylindres]))



X = select(filtered_train, Not(:consommation))
y = filtered_train[:, :consommation]          

train, valid = partition(eachindex(y), 0.8, shuffle=true) 
X_train, y_train = X[train, :], y[train]
X_valid, y_valid = X[valid, :], y[valid]


# https://juliaai.github.io/MLJ.jl/dev/model_stacking/
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree verbosity=0
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees verbosity=0
XGBoostRegressor = @load XGBoostRegressor pkg=XGBoost verbosity=0
KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels verbosity=0
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0

stack = Stack(;metalearner=LinearRegressor(),
                resampling=CV(),
                measures=rmse,
                constant=ConstantRegressor(),
                tree_2=DecisionTreeRegressor(max_depth=2),
                tree_3=DecisionTreeRegressor(max_depth=3),
                evo=EvoTreeRegressor(),
                knn=KNNRegressor(),
                xgb=XGBoostRegressor()
                )

mach = machine(stack, X_train, y_train)

fit!(mach, verbosity=0)

y_pred = collect(predict(mach, X_valid))

println("Stack RMSE: ", rms(y_pred, y_valid))



model = @load LinearRegressor pkg=MLJLinearModels
linear_machine = machine(model(), X_train, y_train)

tree_model = @load RandomForestRegressor pkg=DecisionTree
forest = machine(tree_model(n_trees=100), X_train, y_train)

fit!(linear_machine)
fit!(forest)

y_pred_linear = collect(predict(linear_machine, X_valid))
y_pred_forest = collect(predict(forest, X_valid))

println("Linear RMSE: ", rms(y_pred_linear, y_valid))
println("Random Forest RMSE: ", rms(y_pred_forest, y_valid))


#####################
# Test data prediction
#####################

X_test = one_hot_encode(test_data, categorical_cols)

# linear_machine = machine(model(), X, y)
# forest = machine(tree_model(n_trees=100), X, y)
# fit!(forest)

# y_pred = predict_mode(forest, X_test)


DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree verbosity=0
EvoTreeRegressor = @load EvoTreeRegressor verbosity=0
XGBoostRegressor = @load XGBoostRegressor verbosity=0
KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels verbosity=0
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0

stack = Stack(;metalearner=LinearRegressor(),
                resampling=CV(),
                measures=rmse,
                constant=ConstantRegressor(),
                tree_2=DecisionTreeRegressor(max_depth=2),
                tree_3=DecisionTreeRegressor(max_depth=3),
                evo=EvoTreeRegressor(),
                knn=KNNRegressor(),
                xgb=XGBoostRegressor())

mach = machine(stack, X, y)

evaluate!(mach; resampling=Holdout(), measure=rmse)
println(report(mach).cv_report)
fit!(mach, verbosity=0)

y_pred = collect(predict(mach, X_valid))

n = size(y_pred, 1)
id = 1:n
df_pred = DataFrame(id=id, consommation=y_pred)

# describe(df_pred)

CSV.write("benchmark.csv", df_pred)

