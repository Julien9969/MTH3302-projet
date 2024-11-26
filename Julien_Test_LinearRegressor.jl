using CSV
using DataFrames
using MLJ
using Random
using Statistics

Random.seed!(1234)

train_data = CSV.read("train.csv", DataFrame, decimal=',')
test_data = CSV.read("test.csv", DataFrame, decimal=',')


function remove_outliers(data, column)
    unique_values = unique(data[:, column])
    for value in unique_values
        subset = data[data[:, column] .== value, :]
        if size(subset, 1) == 0
            continue
        end
        q1 = quantile(subset[:, :consommation], 0.25)
        q3 = quantile(subset[:, :consommation], 0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data[:, column] .!= value) .| ((data[:, column] .== value) .& (data[:, :consommation] .>= lower_bound) .& (data[:, :consommation] .<= upper_bound)), :]
    end
    return data
end

filtered_train = copy(train_data)

filtered_train = remove_outliers(filtered_train, :annee);
filtered_train = remove_outliers(filtered_train, :type);
filtered_train = remove_outliers(filtered_train, :nombre_cylindres);
filtered_train = remove_outliers(filtered_train, :transmission);
filtered_train = remove_outliers(filtered_train, :boite);

# describe(filtered_train[:, :consommation])

# filtered_train[:, :annee] = Float64.(filtered_train[:, :annee])


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
# println(schema(filtered_train))
# println(eltype(filtered_train[:, :nombre_cylindres]))


X = select(filtered_train, Not(:consommation))
y = filtered_train[:, :consommation]          

train, test = partition(eachindex(y), 0.8, shuffle=true) 
X_train, y_train = X[train, :], y[train]
X_test, y_test = X[test, :], y[test]


model = @load LinearRegressor pkg=MLJLinearModels
machine_ = machine(model(), X_train, y_train)
fit!(machine_)

# y_pred = predict_mode(machine_, X_test)
y_pred = y_pred = collect(predict(machine_, X_test))


# evaluator = evaluate!(machine_, resampling=CV(shuffle=true), measure=[rms], verbosity=0)
# println("RMSE : ", evaluator.measurement)

println("RMSE : ", rms(y_pred, y_test))
# println("values pred : ", y_pred)

X_test = one_hot_encode(test_data, categorical_cols)

machine_ = machine(model(), X, y)
fit!(machine_)

y_pred = predict_mode(machine_, X_test)

n = size(y_pred, 1)
id = 1:n
df_pred = DataFrame(id=id, consommation=y_pred)

# describe(df_pred)

CSV.write("benchmark.csv", df_pred)

