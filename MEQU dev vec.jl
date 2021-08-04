using LightGraphs
using GraphPlot
using Distributions
using Plots
using StatsPlots
using StatsBase

mutable struct firm
    """
    Agent - firm
    """
    id::Int64
    price::Vector{Float64}
    advertising::Vector{Float64}
    quality::Float64
    quality_variance::Float64
    quality_variance_experience::Float64
    quality_variance_ad::Float64
end

mutable struct customer
    """
    Agent - customer
    """
    id::Int64
    std_reservation_price::Float64
    quality_expectation::Vector{Float64}
    quality_uncertainty::Vector{Float64}
    unit_bought::BitVector
    quality_of_unit_bought::Float64
    received_ad::BitVector
    stock::Int64
    risk::Float64
    history_experience::Vector{BitVector}
    history_wom::Vector{BitVector}
    history_ad::Vector{BitVector}
    loyalty::BitVector
end

function create_network(type::String; num_customers::Int64, num_links::Int64=0, pref_attachment_links::Int64=0)
    """
    Function creating network

    type - def. network constructor type, either random or preferential attachment;
    num_customers - def. problem size, number of agents => number of vertices;
    num_links - def. total number of links, for random network only;
    pref_attachment_links - def. number of each agent's connections, for preferential attachment network only.

    """
    if type == "random"
        return SimpleGraph(num_customers, num_links)
    elseif type == "preferential_attachment"
        return barabasi_albert(num_customers, pref_attachment_links)
    end
end


function create_firms(num_firms::Int64, A::Vector{Float64}, σ_v0::Vector{Float64}, price::Vector{Vector{Float64}}, advertising::Vector{Vector{Float64}}, σ_ϵ::Vector{Float64}, σ_α::Vector{Float64})::Vector{firm}
    """
    Function creating vector of firms

    num_firms - number of firms in the market;
    A - average level of quality of products in the market, common for all brands, prior for each customer;
    σ_v0 - st.dev. of quality level around A, q~N(A,σ_v0), one per brand;
    price - minimal selling price, one per brand
    σ_ϵ - st. dev. of experience signals, one per brand
    σ_α - st. dev. of advertising signals, one per brand

    """
    firms_vector = []
    for s in 1:num_firms
        new_firm = firm(s, #id::Int64
                        price[s],# price::Vector{Float64}
                        advertising[s], #advertising
                        A[s],# quality::Float64
                        σ_v0[s],# quality_variance::Float64
                        σ_ϵ[s], # quality_variance_experience::Float64
                        σ_α[s] # quality_variance_ad::Float64
                        )    # quantity::Int64
        push!(firms_vector, new_firm)
    end
    return firms_vector
end

function create_customers(num_customers::Int64, firms::Vector{firm}, A::Array{Float64}, σ_v0::Vector{Float64} = [1.0], max_stock_period::Int64=7, wealth::Float64=50.0, risk::Float64=0.6)::Vector{customer}
    """
    Function creating vector of customers

    num_customers - number of customers in the market;
    firms - vector of firms;
    A - average level of quality of products in the market, common for all brands, prior for each customer;
    σ_v0 - st.dev. of quality level around A, q~N(A,σ_v0), one per brand;
    max_stock_period - number of days needed for stock to vanish;
    wealth - standard reservation price dominant

    """
    customers_vector = []
    for b in 1:num_customers
        new_customer = customer(b, #id::Int64
                                rand(TriangularDist(0,100,wealth)),# std_reservation_price::Float64
                                A,# quality_expectation::Vector{Float64}
                                σ_v0,# quality_uncertainty::Vector{Float64}
                                falses(length(firms)),# unit_bought::BitVector
                                0.0,# quality_of_unit_bought::Float64
                                falses(length(firms)), # received_ad::BitVector
                                sample(1:max_stock_period), # stock::Int64
                                rand(TriangularDist(0,100,risk)), #risk
                                Vector{BitVector}(), #experience history
                                Vector{BitVector}(), #wom history
                                Vector{BitVector}(),
                                falses(length(firms))) #ad history
        push!(customers_vector, new_customer)
    end
    return customers_vector
end

find_neighbours_positions = function(g; accessibility_step)
    return findall((g .> 0) .& (g .<= accessibility_step))
end

function use_stock(buyer)
    if buyer.stock > 0
        buyer.stock -= 1
    end
    return buyer
end


function choose_best(brands, utility, surplus)

    positive_surplus_choices = findall(surplus .> 0)

    brands = brands[positive_surplus_choices]
    utility = utility[positive_surplus_choices]
    surplus = surplus[positive_surplus_choices]
    
    if length(positive_surplus_choices) > 1
    
        highest_utility_choices = findall(utility .== maximum(utility))
    
        brands = brands[highest_utility_choices]
        utility = utility[highest_utility_choices]
        surplus = surplus[highest_utility_choices]
    
        if length(highest_utility_choices) > 1
    
            highest_surplus_choices = findall(surplus .== maximum(surplus))
    
            brands = brands[highest_surplus_choices]
            utility = utility[highest_surplus_choices]
            surplus = surplus[highest_surplus_choices]
    
            if length(highest_surplus_choices) > 1
    
                chosen_brand = sample(1:length(highest_surplus_choices),1)
    
                brands = brands[chosen_brand]
                utility = utility[chosen_brand]
                surplus = surplus[chosen_brand]
    
            end
    
        end
    
    end

    return Int(brands[1]), Float64(utility[1]), Float64(surplus[1])

end

function customer_choice(buyer; price_current, firms)

    """
    Function simulation customers making decisions

    customers - vector of customers;
    firms - vector of firms;
    max_stock - number of days that stock will last after purchase

    """

    if buyer.stock <= 0
        buyer_wtp = buyer.std_reservation_price .* buyer.quality_expectation
        surplus = buyer_wtp .- price_current
        @assert !any(isnan.(surplus))
        utility = buyer_wtp .- buyer.risk .* buyer.quality_uncertainty
        if any(surplus .>= 0)
            brands = collect(1:length(firms))
            best_product, best_utility, best_surplus = choose_best(brands, utility, surplus)
            @assert length(best_product) == 1
            chosen_firm_bool = falses(length(firms))
            chosen_firm_bool[best_product] = true
            setfield!(buyer, :unit_bought, chosen_firm_bool)
            push!(buyer.history_experience, chosen_firm_bool)
            buyer.stock = max_stock_period
        else
            push!(buyer.history_experience, falses(length(firms)))
        end
    else
        push!(buyer.history_experience, falses(length(firms)))
    end
    return buyer
end

function simulate_quality(buyer; firms, dist)
    e_quality = firms[argmax(buyer.unit_bought)].quality
    var_quality = firms[argmax(buyer.unit_bought)].quality_variance_experience
    prod_qual = e_quality + sample_quality_1dim(dist; quality_variance = var_quality)
    setfield!(buyer, :quality_of_unit_bought, prod_qual)
    return buyer
end


function sample_quality_1dim(dist::String; quality_variance::Float64=0)

    """
    Function sampling bias in estimate of quality, one dimensional

    dist - distribution of sampling, uniform or trimmed normal
    quality_variance - variance of quality

    """


    if dist == "uniform"
        qv = minimum([quality_variance, 1/3])
        qv = sqrt(12 * qv) / 2
        return rand(Uniform(-qv, qv))
    elseif dist == "trimmed_normal"
        qv = rand(Normal(0, quality_variance))
        #qv = minimum([qv,1])
        #qv = maximum([qv,-1])
        return qv
    end
end

function ad_receive(buyer; ad_intensity)

    rec_ad = BitVector(rand.(Binomial.(1, ad_intensity)))
    setfield!(buyer, :received_ad, rec_ad)
    push!(buyer.history_ad, rec_ad)
    return buyer

end

function clear_previous_step(buyer; firms)
    setfield!(buyer, :unit_bought, falses(length(firms)))
    setfield!(buyer, :received_ad, falses(length(firms)))
    setfield!(buyer, :received_ad, falses(length(firms)))
    return buyer
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



function consumption_surprise(buyer; firms)
    surprise_my_prod_qual = zeros(Float64, length(firms))

    if any(buyer.unit_bought)
        surprise_my_prod_qual = buyer.unit_bought .* (buyer.quality_of_unit_bought .- buyer.quality_expectation)
    end
    return surprise_my_prod_qual
end

function wom_surprise(buyer, neighbours_id; customers, firms)
    
    my_neighbours = customers[neighbours_id]

    surprise_neigh_prod_qual = zeros(Float64, length(firms))

    neigh_prod_quality = reduce(+, peer_observation_quality.(my_neighbours))
    neigh_prod_quantity = reduce(+, peer_observation_quantity.(my_neighbours))

    surprise_neigh_prod_quality = neigh_prod_quality ./ neigh_prod_quantity - buyer.quality_expectation
    surprise_neigh_prod_quality[isnan.(surprise_neigh_prod_quality)] .= 0

    return surprise_neigh_prod_quality, neigh_prod_quantity

end

function peer_observation_quality(neigh_buyer)

    neigh_prod_quality = neigh_buyer.unit_bought .* neigh_buyer.quality_of_unit_bought

    return neigh_prod_quality
end

function peer_observation_quantity(neigh_buyer)

    neigh_prod_buy = neigh_buyer.unit_bought

    return neigh_prod_buy
end

function ad_surprise(buyer; firms, dist)
    e_quality = getfield.(firms, :quality)
    var_quality = getfield.(firms, :quality_variance_ad)
    sampled_ad = e_quality .+ sample_quality_ndim(dist, quality_variance = var_quality)
    surprise_ad_qual = buyer.received_ad .* (sampled_ad .- buyer.quality_expectation)
    return surprise_ad_qual
end

function calculate_memory(iter, buyer_memory)

    k = buyer_memory

    if iter < buyer_memory
        k = iter
    end

    return k

end

function calculate_kalman(uncertainty; signal_variance)
    kalman_gain = uncertainty ./ (uncertainty .+ signal_variance)
    return kalman_gain
end

function calculate_uncertainty(buyer; backward_memory, λ_ind, λ_wom, λ_ad, firms_experience_variance, firms_ad_variance, base_uncertainty)

    memory_history_experience = reduce(+,buyer.history_experience[(end-backward_memory+1):end])
    memory_history_wom = reduce(+,buyer.history_wom[(end-backward_memory+1):end])
    memory_history_ad = reduce(+,buyer.history_ad[(end-backward_memory+1):end])

    experience_uncertainty = λ_ind * memory_history_experience ./ firms_experience_variance
    wom_uncertainty = λ_wom * memory_history_wom ./ firms_experience_variance # since rely on mean, its as it was a single signal
    ad_uncertainty = λ_ad * memory_history_ad ./ firms_ad_variance

    new_uncertainty = 1 ./ (base_uncertainty .+ experience_uncertainty .+ wom_uncertainty .+ ad_uncertainty)

    return new_uncertainty

end

function push_wom(buyer, wom_quantity)
    push!(buyer.history_wom, BitVector(sign.(wom_quantity)))
    return buyer
end


function bayesian_updating(customers, firms, my_neighbours_ids, dist, iter, buyer_memory, λ_ind,λ_wom, λ_ad)

    firms_base_variance = getfield.(firms, :quality_variance).^2
    firms_experience_variance = getfield.(firms, :quality_variance_experience).^2
    firms_ad_variance = getfield.(firms, :quality_variance_ad).^2
    base_uncertainty = 1 ./ firms_base_variance


    experience_surprise_quality = consumption_surprise.(customers; firms)

    wom_surprise_quality_and_quantity = wom_surprise.(customers, my_neighbours_ids; customers, firms)
    wom_surprise_quality = getindex.(wom_surprise_quality_and_quantity, 1)
    wom_quantity = getindex.(wom_surprise_quality_and_quantity, 2)
    customers = push_wom.(customers, wom_quantity)
    ad_surprise_quality = ad_surprise.(customers; firms, dist)

    backward_memory = calculate_memory(iter, buyer_memory)

    new_uncertainty = calculate_uncertainty.(customers; backward_memory,λ_ind,λ_wom, λ_ad, firms_experience_variance, firms_ad_variance, base_uncertainty)
    
    setfield!.(customers, :quality_uncertainty, new_uncertainty)
    
    experience_kalman_gain = calculate_kalman.(new_uncertainty; signal_variance = firms_experience_variance)
    ad_kalman_gain = calculate_kalman.(new_uncertainty; signal_variance = firms_ad_variance)
    wom_kalman_gain = calculate_kalman.(new_uncertainty; signal_variance = firms_experience_variance)

    experience_learning = calculate_learning.(getfield.(customers, :unit_bought), experience_kalman_gain, experience_surprise_quality)
    ad_learning = calculate_learning.(getfield.(customers, :received_ad), ad_kalman_gain, ad_surprise_quality)
    wom_learning = calculate_learning.(map(x -> x .> 0, wom_quantity), wom_kalman_gain, wom_surprise_quality)

    new_quality_expectation = calculate_new_quality_expectation.(getfield.(customers, :quality_expectation), experience_learning, wom_learning, ad_learning)

    setfield!.(customers, :quality_expectation, new_quality_expectation)
    
    return customers

end

calculate_learning(a,b,c) = a .* b .* c

calculate_new_quality_expectation(a,b,c,d) = a .+ b .+ c .+ d

function inverse_gdistances(i; network)
    return gdistances(network, i)
end

function simulate(num_firms, a, σ_v0, prices, advertising, σ_ϵ, σ_α, num_customers, max_stock_period, wealth, risk, network_type, num_links, pref_attachment_links, accessibility_step, max_iter)

    # preparado

    firms = create_firms(num_firms, a, σ_v0, prices, advertising, σ_ϵ, σ_α) # done
    customers = create_customers(num_customers, firms, a, σ_v0, max_stock_period, wealth, risk) # done
    network = create_network(network_type; num_customers = num_customers, num_links = num_links, pref_attachment_links = pref_attachment_links) # done
    @time neighbours_distances = inverse_gdistances.(1:length(customers); network) # done
    my_neighbours_ids = find_neighbours_positions.(neighbours_distances; accessibility_step) # done

    # sempre

    for iter in 1:max_iter

        price_current = getindex.(getfield.(firms, :price), iter) # done
        ad_intensity = getindex.(getfield.(firms, :advertising), iter) # done
        customers = clear_previous_step.(customers; firms) # done
        customers = use_stock.(customers)  # done
        @time customers = customer_choice.(customers; price_current, firms) # to improve
        customers = simulate_quality.(customers; firms, dist)
        @time customers = ad_receive.(customers; ad_intensity)
        @time customers = bayesian_updating(customers, firms,  my_neighbours_ids, dist, iter, buyer_memory,  λ_ind,λ_wom, λ_ad) # to improve


    end

    return customers

end

ENV["JULIA_NUM_THREADS"] = 6
Threads.nthreads()

@time simulate(NUM_FIRMS, A,Σ_V0, PRICES, ADVERTISING, Σ_Ε, Σ_Α, NUM_CUSTOMERS, MAX_STOCK_PERIOD, WEALTH, RISK, NETWORK_TYPE, NUM_LINKS, PREF_ATTACHMENT_LINKS, ACCESSIBILITY_STEP, 1)

#%% SANDBOX ###
`
num_firms = NUM_FIRMS = 6
num_customers = NUM_CUSTOMERS = 200
network_type = NETWORK_TYPE = "preferential_attachment" # type of network between agents
num_links = NUM_LINKS = 200 # number of links, for random graphs only
pref_attachment_links = PREF_ATTACHMENT_LINKS = 2 # number of neighbours per agent, for barabasi-albert only

dist = DIST = "trimmed_normal" # distribution of quality
accessibility_step = ACCESSIBILITY_STEP = 1 # distance from agent to agent, defining neighbourhood

λ_ind = Λ_IND = 1. # individual learning

λ_wom = Λ_WOM = 1. # social learning
λ_ad = Λ_AD = 1.

ad_intensity = AD_INTENSITY = fill(0.0,10)[1:NUM_FIRMS]

a = A = fill(1., NUM_FIRMS)
σ_v0 = Σ_V0 = fill(0.8, 10)[1:NUM_FIRMS]
σ_α = Σ_Α = fill(0.7*5, 10)[1:NUM_FIRMS]
σ_ϵ = Σ_Ε = fill(0.9*5, 10)[1:NUM_FIRMS]
max_iter = MAX_ITER = 365
prices = PRICES = [rand(Normal(50, 1), MAX_ITER) for x in 1:NUM_FIRMS]
PRICES[1][50:75] .= 46
advertising = ADVERTISING = [rand(Uniform(0,0.1), MAX_ITER) for x in 1:NUM_FIRMS]
ADVERTISING[3][10:35] .= 0.40

ADVERTISING = fill(fill(0.0, MAX_ITER), NUM_FIRMS)

max_stock_period = MAX_STOCK_PERIOD = 1
wealth = WEALTH = 10.0
risk = RISK = 8.0
method = METHOD = "markovitz"
buyer_memory = BUYER_MEMORY = 150
min_loyal_period = MIN_LOYAL_PERIOD = 2000
loyalty_buying = LOYALTY_BUYING = false
only_loyalty_wom = ONLY_LOYALTY_WOM = false



`