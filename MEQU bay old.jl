function calculate_price_and_traded_units(customers::Vector{customer}, firms::Vector{firm}; max_stock_period::Int64 = 7, iter::Int64, method::String="max_surplus", loyalty_buying::Bool=true)

    """
    Function simulation customers making decisions

    customers - vector of customers;
    firms - vector of firms;
    max_stock - number of days that stock will last after purchase

    """

    setfield!.(customers, :unit_bought, fill(falses(length(firms)), length(customers)))
    setfield!.(customers, :received_ad, fill(falses(length(firms)), length(customers)))

    price_plan = getfield.(firms, :price)

    if method == "max_surplus"

        for buyer in customers
            push!(buyer.history_experience, falses(length(firms)))
            buyer.stock -= 1
            if buyer.stock <= 0
                surplus = buyer.reservation_price .- getindex.(price_plan, iter)
                @assert !any(isnan.(surplus))
                setfield!(buyer, :surplus, surplus)
                if any(surplus .>= 0)
                    if (sum(buyer.loyalty) == 1) & loyalty_buying
                        chosen_firm = argmax(buyer.loyalty)
                    else
                        chosen_firm = argmax(surplus)
                    end
                    @assert length(chosen_firm) == 1
                    chosen_firm_bool = falses(length(firms))
                    chosen_firm_bool[chosen_firm] = true
                    setfield!(buyer, :unit_bought, chosen_firm_bool)
                    buyer.history_experience[end] = chosen_firm_bool
                    buyer.stock = max_stock_period
                    @assert buyer.stock > 0
                end
            end
        end

    elseif method == "max_value"

        for buyer in customers
            push!(buyer.history_experience, falses(length(firms)))
            buyer.stock -= 1
            if buyer.stock <= 0
                values = buyer.reservation_price
                surplus = values .- getindex.(price_plan, iter)
                @assert !any(isnan.(values))
                @assert !any(isnan.(surplus))
                setfield!(buyer, :surplus, surplus)
                if any(surplus .>= 0)
                    brands = collect(1:length(firms))
                    brands = brands[surplus .>= 0]
                    values = values[surplus .>= 0]
                    if (sum(buyer.loyalty) == 1) & loyalty_buying
                        chosen_firm = argmax(buyer.loyalty)
                    else
                        chosen_firm = argmax(values)
                    end
                    @assert length(chosen_firm) == 1
                    chosen_firm_bool = falses(length(firms))
                    chosen_firm_bool[chosen_firm] = true
                    setfield!(buyer, :unit_bought, chosen_firm_bool)
                    buyer.history_experience[end] = chosen_firm_bool
                    buyer.stock = max_stock_period
                    @assert buyer.stock > 0
                end
            end
        end

    elseif method == "max_value_with_uncertainty" #🍎

        for buyer in customers
            push!(buyer.history_experience, falses(length(firms)))
            buyer.stock -= 1
            if buyer.stock <= 0
                values = buyer.reservation_price
                uncertainty = buyer.quality_uncertainty.^2
                utility = values - 0.25(buyer.risk .* values.^2 + buyer.risk .* uncertainty .* buyer.std_reservation_price)
                surplus = values .- getindex.(price_plan, iter)
                @assert !any(isnan.(values))
                @assert !any(isnan.(surplus))
                setfield!(buyer, :surplus, surplus)
                if any(surplus .>= 0)
                    brands = collect(1:length(firms))
                    brands = brands[surplus .>= 0]
                    utility = utility[surplus .>= 0]
                    if (sum(buyer.loyalty) == 1) & loyalty_buying
                        chosen_firm = argmax(buyer.loyalty)
                    else
                        chosen_firm = brands[argmax(utility)]
                    end
                    @assert length(chosen_firm) == 1
                    chosen_firm_bool = falses(length(firms))
                    chosen_firm_bool[chosen_firm] = true
                    setfield!(buyer, :unit_bought, chosen_firm_bool)
                    buyer.history_experience[end] = chosen_firm_bool
                    buyer.stock = max_stock_period

                end
            end
        end

    elseif method == "max_surplus_with_uncertainty" #🍎

        for buyer in customers
            push!(buyer.history_experience, falses(length(firms)))
            buyer.stock -= 1
            if buyer.stock <= 0
                values = buyer.reservation_price
                surplus = buyer.reservation_price .- getindex.(price_plan, iter)
                uncertainty = buyer.quality_uncertainty.^2
                utility = surplus - 0.25(buyer.risk .* values.^2 + buyer.risk .* uncertainty .* buyer.std_reservation_price)
                @assert !any(isnan.(surplus))
                setfield!(buyer, :surplus, surplus)
                if any(surplus .>= 0)
                    brands = collect(1:length(firms))
                    brands = brands[surplus .>= 0]
                    utility = utility[surplus .>= 0]
                    if (sum(buyer.loyalty) == 1) & loyalty_buying
                        chosen_firm = argmax(buyer.loyalty)
                    else
                        chosen_firm = brands[argmax(utility)]
                    end
                    @assert length(chosen_firm) == 1
                    chosen_firm_bool = falses(length(firms))
                    chosen_firm_bool[chosen_firm] = true
                    setfield!(buyer, :unit_bought, chosen_firm_bool)
                    buyer.history_experience[end] = chosen_firm_bool
                    buyer.stock = max_stock_period

                end
            end
        end

    elseif method == "markovitz"

        for buyer in customers

            push!(buyer.history_experience, falses(length(firms)))
            buyer.stock -= 1
            if buyer.stock <= 0
                values = buyer.reservation_price
                surplus = buyer.reservation_price .- getindex.(price_plan, iter)
                uncertainty = buyer.quality_uncertainty.^2
                utility = values .- buyer.risk .* buyer.quality_uncertainty
                @assert !any(isnan.(surplus))
                setfield!(buyer, :surplus, surplus)
                if any(surplus .>= 0)
                    brands = collect(1:length(firms))
                    brands = brands[surplus .>= 0]
                    utility = utility[surplus .>= 0]
                    if (sum(buyer.loyalty) == 1) & loyalty_buying
                        chosen_firm = argmax(buyer.loyalty)
                    else
                        chosen_firm = brands[argmax(utility)]
                    end
                    @assert length(chosen_firm) == 1
                    chosen_firm_bool = falses(length(firms))
                    chosen_firm_bool[chosen_firm] = true
                    setfield!(buyer, :unit_bought, chosen_firm_bool)
                    buyer.history_experience[end] = chosen_firm_bool
                    buyer.stock = max_stock_period
    
                end
            end
            
        end

    end

    return customers
end

function sample_quality_ndim(dist::String; quality_variance::Vector{Float64})
    """
    Function sampling bias in estimate of quality, vectorized

    dist - distribution of sampling, uniform or trimmed normal
    quality_variance - variance of quality

    """
    if dist == "uniform"
        qv = min.(quality_variance, 1/3)
        qv = sqrt.(12 * qv) ./ 2
        return rand.(Uniform.(-qv, qv))
    elseif dist == "trimmed_normal"
        qv = rand.(Normal.(0, quality_variance))
        #qv = min.(qv,2)
        #qv = max.(qv,-2)
        return qv
    end
end