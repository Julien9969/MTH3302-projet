using GLM
using MLJ
using MLJLinearModels
using StatsModels

function remove_outliers(data)
    model = lm(@formula(y ~ x), data) # to get encoding

    X = modelmatrix(model)[:,2]
    H = X * inv(X' * X) * X'
    leverage = diag(H)

    predictions =  StatsModels.predict(model)
    residuals = y .- predictions

    student_residuals = residuals ./ sqrt.(1 .- leverage)
    treshold = 3
    outliers = abs.(student_residuals) .> treshold
    data = data[.!outliers, :]

    return data
end

function remove_outliers(data, features::Vector{Symbol}, target::Symbol=:y)
    formula = Term(target) ~ sum(Term(Symbol(feature)) for feature in features)

    model = lm(formula, data) # to get encoding

    X = modelmatrix(model)[:,2]
    H = X * inv(X' * X) * X'
    leverage = diag(H)

    predictions =  StatsModels.predict(model)
    residuals = y .- predictions

    student_residuals = residuals ./ sqrt.(1 .- leverage)
    treshold = 3
    outliers = abs.(student_residuals) .> treshold
    data = data[.!outliers, :]

    return data
end

function create_model(data, features::Vector{Symbol}, target::Symbol=:y)
    formula = Term(target) ~ sum(Term(Symbol(feature)) for feature in features)

    model = lm(formula, data)

    return model
end

function create_model(data, target::Symbol=:y)
    formula = Term(target) ~ sum(Term(Symbol(feature)) for feature in filter(x -> Symbol(x) != target, names(data)))

    model = lm(formula, data)

    return model
end

function one_hot_encode(data)
    for feature in names(data)
        if eltype(data[:, feature]) <: AbstractString
            coerce!(data, feature => Multiclass)
        else
            # data[:, feature] = Float64.(data[:, feature])
        end
    end

    one_hot = machine(OneHotEncoder(), data)
    fit!(one_hot, verbosity=0)
    data = MLJ.transform(one_hot, data)
    return data
end

function ridge_machine(λ, X, y)
    RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels verbosity=0 
    model = RidgeRegressor(lambda = λ) 
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)

    return mach
end