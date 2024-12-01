using GLM
using MLJ
using StatsModels
using Turing
using Distributions


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

function encode(data, categorical_features, continuous_features, ordinal_features)
    for feature in categorical_features
        coerce!(data, feature => Multiclass)
    end

    mach = machine(OneHotEncoder(), data)
    fit!(mach, verbosity=0)
    data = MLJ.transform(mach, data)

    for feature in continuous_features
        coerce!(data, feature => MLJ.Continuous)
    end

    mach = machine(ContinuousEncoder(), data)
    fit!(mach, verbosity=0)
    data = MLJ.transform(mach, data)

    for feature in ordinal_features
        data[!, feature] = categorical(data[!, feature], ordered=true)
    end

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

        score = mean(cv_result.measurement)

        if score < best_score
            best_score = score
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

        score = mean(cv_result.measurement)

        if score < best_score
            best_score = score
            λ̂ = lambda
        end
    end

    println(λ̂)
    mach = machine(LassoRegressor(lambda=λ̂) , X, y)
    fit!(mach, verbosity=0)

    return mach
end

function elastic_net_regression_cv(X, y, k_folds=5)
    ElasticNetRegressor = @load ElasticNetRegressor pkg=MLJLinearModels verbosity=0

    lambdas = 10 .^ LinRange(-4, 0, 25)
    gammas = 10 .^ LinRange(-4, 0, 25)

    cv = CV(nfolds=k_folds)

    best_score = Inf
    λ̂ = nothing
    γ̂ = nothing

    for lambda in lambdas
        for gamma in gammas
            model = ElasticNetRegressor(lambda=lambda, gamma=gamma)
            mach = machine(model, X, y)
            cv_result = evaluate!(mach, resampling=cv, verbosity=0)

            score = mean(cv_result.measurement)

            if score < best_score
                best_score = score
                λ̂ = lambda
                γ̂ = gamma
            end
        end
    end

    mach = machine(ElasticNetRegressor(lambda=λ̂, gamma=γ̂), X, y)
    fit!(mach, verbosity=0)

    # return λ̂, γ̂
    return mach
end

@model function bayesian_regression(X, y)
    features = names(X)
    n = size(X, 1)
    predictors = size(X, 2)

    β = zeros(predictors)
    for i in 1:predictors
        if features[i] == "cylindree"
            β[i] ~ LogNormal(5.981630883156576, 3.6476067440594417)
        elseif features[i] == "nombre_cylindres"
            β[i] ~ Normal(5.479391634980988, 1.7817038897119009)
        else 
            β[i] ~ Laplace(0, 1)
        end
    end
    α ~ Normal(mean(y), std(y))

    σ² ~ InverseGamma(2, 1)

    μ = α .+ Matrix(X) * β
    Σ = σ² * I

    return y ~ MvNormal(μ, Σ)
end

# https://turinglang.org/docs/tutorials/05-linear-regression/
function bayesian_prediction(chain, X)
    params = get_params(chain[50:end, :, :])

    α̂ = params.α
    β̂ = reduce(hcat, params.β)
    ŷ_dists = α̂' .+ Matrix(X) * β̂'

    return [mean(ŷ_dists[i, :]) for i in 1:size(ŷ_dists, 1)]
end
