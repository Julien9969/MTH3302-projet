using GLM
using MLJ
using StatsModels
using Turing

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

    model = lm(formula, data)

    y = data[:, target]
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

function mean_impute_outliers(data, features::Vector{Symbol}, target::Symbol=:y)
    formula = Term(target) ~ sum(Term(Symbol(feature)) for feature in features)

    model = lm(formula, data)

    y = data[:, target]
    X = modelmatrix(model)[:,2]
    H = X * inv(X' * X) * X'
    leverage = diag(H)

    predictions = StatsModels.predict(model)
    residuals = y .- predictions

    student_residuals = residuals ./ sqrt.(1 .- leverage)
    treshold = 3
    outliers = abs.(student_residuals) .> treshold
    for feature in features
        if eltype(data[:, feature]) <: AbstractString || eltype(data[:, feature]) <: CategoricalValue
            groups = groupby(data, feature)
            for group in groups
                mean_value = mean(group[:, target])

                data[outliers .& (data[!, feature] .== group[1, feature]), target] .= mean_value
            end
        elseif eltype(data[:, feature]) <: AbstractFloat
            df = DataFrame(x = data[!, feature], y = data[!, target])
            mean_value = mean(df[.!outliers, :y])

            data[outliers, target] .= mean_value
        end
    end

    return data
end

function create_model(data, features::Vector{Symbol}, target::Symbol=:y)
    formula = Term(target) ~ sum(Term(feature) for feature in filter(x -> x != target, features))

    model = lm(formula, data)

    return model
end

function create_model(data, target::Symbol=:y)
    formula = Term(target) ~ sum(Term(Symbol(feature)) for feature in filter(x -> Symbol(x) != target, names(data)))

    model = lm(formula, data)

    return model
end

function one_hot_encode(data, features)
    for feature in features
        if eltype(data[:, feature]) <: AbstractString
            coerce!(data, feature => Multiclass)
        end
    end

    mach = machine(OneHotEncoder(), data)
    fit!(mach, verbosity=0)
    data = MLJ.transform(mach, data)

    return data
end

function linear_regression(X, y)
    LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0 
    linear_regressor = LinearRegressor()

    mach = machine(linear_regressor, X, y)
    fit!(mach, verbosity=0)

    return mach
end

function ridge_regression_cv(X, y, k_folds=5)
    RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels verbosity=0 
    lambdas = 10 .^ LinRange(-4, 4, 100)
    cv = CV(nfolds=k_folds)

    best_score = Inf
    λ̂ = nothing

    for lambda in lambdas
        model = RidgeRegressor(lambda=lambda)
        mach = machine(model, X, y)

        cv_result = evaluate!(mach, resampling=cv, verbosity=0)

        mean_score = mean(cv_result.measurement)

        if mean_score < best_score
            best_score = mean_score
            λ̂ = lambda
        end
    end

    mach = machine(RidgeRegressor(lambda = λ̂) , X, y)
    fit!(mach, verbosity=0)

    return mach
end

function lasso_regression_cv(X, y, k_folds=5)
    LassoRegressor = @load LassoRegressor pkg=MLJLinearModels verbosity=0 
    lambdas = 10 .^ LinRange(-4, 4, 100)
    cv = CV(nfolds=k_folds)

    best_score = Inf
    λ̂ = nothing

    for lambda in lambdas
        model = LassoRegressor(lambda=lambda)
        mach = machine(model, X, y)

        cv_result = evaluate!(mach, resampling=cv, verbosity=0)

        mean_score = mean(cv_result.measurement)

        if mean_score < best_score
            best_score = mean_score
            λ̂ = lambda
        end
    end

    mach = machine(LassoRegressor(lambda=λ̂) , X, y)
    fit!(mach, verbosity=0)

    return mach
end

function elastic_net_regression_cv(X, y, k_folds=5)
    ElasticNetRegressor = @load ElasticNetRegressor pkg=MLJLinearModels verbosity=0

    lambdas = 0.0:0.1:1.0
    gammas = 0.0:0.1:1.0

    cv = CV(nfolds=k_folds)

    best_score = Inf
    λ̂ = nothing
    γ̂ = nothing

    for lambda in lambdas
        for gamma in gammas
            model = ElasticNetRegressor(lambda=lambda, gamma=gamma)
            mach = machine(model, X, y)
            cv_result = evaluate!(mach, resampling=cv, verbosity=0)

            mean_score = mean(cv_result.measurement)

            if mean_score < best_score
                best_score = mean_score
                λ̂ = lambda
                γ̂ = gamma
            end
        end
    end

    mach = machine(ElasticNetRegressor(lambda=λ̂, gamma=γ̂), X, y)
    fit!(mach, verbosity=0)

    return mach
end

@model function bayesian_regression(X, y)
    predictors = size(X, 2)
    α ~ Normal(0, 10)

    β ~ filldist(TDist(3), predictors)
    # β ~ MvNormal(predictors, 10)
    # β = zeros(predictors)
    # for i in 1:predictors
    #     β[i] ~ Laplace(0, 1)
    # end
    σ² ~ InverseGamma(2, 1)

    μ = α .+ X * β
    Σ = sqrt(σ²) * I

    return y ~ MvNormal(μ, Σ)
end

# https://turinglang.org/docs/tutorials/05-linear-regression/
function bayesian_prediction(chain, X)
    params = get_params(chain[200:end, :, :])

    α̂ = params.α
    β̂ = reduce(hcat, params.β)
    ŷ_dists = α̂' .+ X * β̂'

    return [mean(ŷ_dists[i, :]) for i in 1:size(ŷ_dists, 1)]
end
