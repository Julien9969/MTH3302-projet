using GLM

function remove_outliers(x)
    data = DataFrame(y = y, x = x);
    model = lm(@formula(y ~ x), data) # to get encoding

    X = modelmatrix(model)[:,2]
    H = X * inv(X' * X) * X'
    leverage = diag(H)

    predictions = predict(model)
    residuals = y .- predictions

    student_residuals = residuals ./ sqrt.(1 .- leverage)
    treshold = 3
    outliers = abs.(student_residuals) .> treshold
    data = data[.!outliers, :]

    return data
end

function create_model(data)
    model = lm(@formula(y ~ x), data)

    return model
end

function get_coefficient_of_determination(model, data)
    ȳ = mean(data.y)
    ŷ = predict(model)

    SST = sum((data.y .- ȳ).^2)
    SSR = sum((data.y .-  ŷ).^2)
    R² = 1 - (SSR / SST)

    return R²
end
