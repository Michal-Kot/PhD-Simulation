using LightGraphs
using GraphPlot
using Distributions
using Plots
using StatsPlots
using StatsBase

"""

DEVS:

-> extend for J products 👌
-> add bayesian updating 👌
-> add customers features
-> add sampling customers from population
-> add price promotions 👌
-> add utility 👌

--> 2021020
-> small scale solution, analytic & abm
-> add asserts, to check code 👌
-> write down assumptions & equations in overleaf, ref. to code
-> write down slide beamer with main equations
-> think about framework to describe model, e.g. Macal&North

--> things to consider
-> max surplus? or max exp. res. price? 👌
-> moving memory of customers 👌

"""

mutable struct firm
    """
    Agent - firm
    """
    id::Int64
    price::Vector{Float64}
    advertising::Vector{Float64}
    surplus::Float64
    quality::Float64
    quality_variance::Float64
    quality_variance_experience::Float64
    quality_variance_ad::Float64
    quantity::Int64
end

mutable struct customer
    """
    Agent - customer
    """
    id::Int64
    std_reservation_price::Float64
    quality_expectation::Vector{Float64}
    quality_uncertainty::Vector{Float64}
    reservation_price::Vector{Float64}
    unit_bought::BitVector
    quality_of_unit_bought::Float64
    surplus::Vector{Float64}
    received_ad::BitVector
    stock::Int64
    risk::Float64
    history_experience::Vector{BitVector}
    history_wom::Vector{BitVector}
    history_ad::Vector{BitVector}
    loyalty::BitVector
end

# FUNCTIONS

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
                        0.0,# surplus::Float64
                        A[s],# quality::Float64
                        σ_v0[s],# quality_variance::Float64
                        σ_ϵ[s], # quality_variance_experience::Float64
                        σ_α[s],# quality_variance_ad::Float64
                        0)    # quantity::Int64
        push!(firms_vector, new_firm)
    end
    return firms_vector
end

function choose_distr(x)
    if x == 0.0
        return 0.0
    elseif x == 1.0
        return 1.0
    else
        return rand(TriangularDist(0,1,x))
    end
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
                                fill(b * 1.0, length(firms)),# reservation_price::Vector{Float64}
                                falses(length(firms)),# unit_bought::BitVector
                                0.0,# quality_of_unit_bought::Float64
                                fill(0.0, length(firms)),    # surplus::Vector{Float64}
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

    if method == "markovitz"

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

function exchange_goods(customers::Vector{customer}, firms::Vector{firm}; dist)
    """
    Function simulating quality of purchased goods

    customers - vector of customers;
    firms - vector of firms;
    dist - distribution of sampling, uniform or trimmed normal

    """
    customers = simulate_quality.(customers; firms=firms, dist=dist)

    return customers

end





function receive_advertising(customers::Vector{customer}, firms::Vector{firm}; intensity::Vector{Float64}, dist::String, iter::Int64)
    """
    Function simulating received advertising

    customers - vector of customers;
    firms - vector of firms;
    intensity - opportunity to see ad
    dist - distribution of sampling, uniform or trimmed normal

    """

    intensity = getindex.(getfield.(firms, :advertising), iter)

    for buyer in customers
        rec_ad = BitVector(rand.(Binomial.(1, intensity)))
        setfield!(buyer, :received_ad, rec_ad)
        push!(buyer.history_ad, rec_ad)
    end

    return customers

end

function update_quality_expectation_BAYESIAN(customers::Vector{customer}, firms::Vector{firm}, network; accessibility_step::Int64=1, λ_ind::Float64=0.0, λ_wom::Float64=0.0, λ_ad::Float64=0.0, dist::String="trimmed_normal", iter::Int64=1, buyer_memory::Int64=1, only_loyalty_wom::Bool=false)
    """
    Function simulating how signals affect quality perception. Customers learn in Bayesian manner.

    customers - vector of customers;
    firms - vector of firms;
    network - network connecting customers
    accessibility_step - def. neighbourhood, 1 - only direct neighbours, 2 - up to direct neighbours of my neighbours etc.
    λ_ind - learning rate, individual experience
    λ_wom - learning rate, neighbours experience
    λ_ad - learning rate, advertising exposure
    dist - distribution of sampling, uniform or trimmed normal

    """

    firms_base_variance = getfield.(firms, :quality_variance).^2
    firms_experience_variance = getfield.(firms, :quality_variance_experience).^2
    firms_ad_variance = getfield.(firms, :quality_variance_ad).^2

    base_uncertainty = 1 ./ firms_base_variance

    signals = []
    learning_buyer = []

    for buyer in customers

        # signal, based on individual experience

        surprise_my_prod_qual = zeros(Float64, length(firms))

        if any(buyer.unit_bought)
            surprise_my_prod_qual = buyer.unit_bought .* (buyer.quality_of_unit_bought .- buyer.quality_expectation)
        end

        # signal, based on neighbours' experience

        push!(buyer.history_wom, falses(length(firms)))

        my_id = buyer.id
        my_neighbours_distance = gdistances(network, my_id)
        my_neighbours = customers[(my_neighbours_distance .<= accessibility_step) .& (my_neighbours_distance .> 0)]

        neigh_prod_quality = zeros(Float64, length(firms))
        neigh_prod_buy = zeros(Float64, length(firms))
        surprise_neigh_prod_qual = zeros(Float64, length(firms))

        for neigh_buyer in my_neighbours
            if any(neigh_buyer.unit_bought)
                if only_loyalty_wom
                    neigh_prod_quality += neigh_buyer.loyalty .* neigh_buyer.unit_bought .* neigh_buyer.quality_of_unit_bought
                    neigh_prod_buy += neigh_buyer.loyalty .* neigh_buyer.unit_bought
                else
                    neigh_prod_quality += neigh_buyer.unit_bought .* neigh_buyer.quality_of_unit_bought
                    neigh_prod_buy += neigh_buyer.loyalty .* neigh_buyer.unit_bought
                end
            end
        end

        if any(neigh_prod_buy .> 0)
            surprise_neigh_prod_qual = neigh_prod_quality ./ neigh_prod_buy - buyer.quality_expectation
            surprise_neigh_prod_qual[isnan.(surprise_neigh_prod_qual)] .= 0
            buyer.history_wom[end] = BitVector(sign.(neigh_prod_buy))
        end

        # signal, based on advertising

        surprise_ad_qual = zeros(Float64,length(firms))

        if any(buyer.received_ad)
            e_quality = getfield.(firms, :quality)
            var_quality = getfield.(firms, :quality_variance_ad)
            sampled_ad = e_quality .+ sample_quality_ndim(dist, quality_variance = var_quality)
            surprise_ad_qual = buyer.received_ad .* (sampled_ad .- buyer.quality_expectation)
        end

        # variance recalculation

        k = buyer_memory

        if iter < k
            k = iter
        end

        memory_history_experience = reduce(+,buyer.history_experience[(end-k+1):end])
        memory_history_wom = reduce(+,buyer.history_wom[(end-k+1):end])
        memory_history_ad = reduce(+,buyer.history_ad[(end-k+1):end])

        experience_uncertainty = λ_ind * memory_history_experience ./ firms_experience_variance
        wom_uncertainty = λ_wom * memory_history_wom ./ firms_experience_variance # since rely on mean, its as it was a single signal
        ad_uncertainty = λ_ad * memory_history_ad ./ firms_ad_variance

        new_uncertainty = 1 ./ (base_uncertainty .+ experience_uncertainty .+ wom_uncertainty .+ ad_uncertainty)

        #@assert all(new_uncertainty .< buyer.quality_uncertainty)

        setfield!(buyer, :quality_uncertainty, new_uncertainty)

        # expectation recalculation

        experience_kalman = new_uncertainty ./ (new_uncertainty .+ firms_experience_variance)
        wom_kalman = new_uncertainty ./ (new_uncertainty .+ firms_experience_variance) # 🍎 check if additional feature - observation bias to be introduced
        ad_kalman = new_uncertainty ./ (new_uncertainty .+ firms_ad_variance)

        experience_learning = buyer.unit_bought .* experience_kalman .* surprise_my_prod_qual
        wom_learning = sign.(neigh_prod_buy) .* wom_kalman .* surprise_neigh_prod_qual
        ad_learning = buyer.received_ad .* ad_kalman .* surprise_ad_qual

         @assert all(-2 .<= experience_learning .<= 2)
         @assert all(-2 .<= wom_learning .<= 2)
         @assert all(-2 .<= ad_learning .<= 2)

        new_quality_expectation = buyer.quality_expectation .+ experience_learning .+ wom_learning .+ ad_learning

        setfield!(buyer, :quality_expectation, new_quality_expectation)
        setfield!(buyer, :reservation_price, new_quality_expectation .* getfield(buyer, :std_reservation_price))

        push!(learning_buyer, new_quality_expectation)

    end

    return customers, learning_buyer

end

function calculate_loyalty(customers::Vector{customer}, min_loyal_period::Int64, iter ::Int64)
    if iter >= min_loyal_period
        for buyer in customers
            if sum(buyer.loyalty) == 0
                buying_last_k = reduce(+,buyer.history_experience[(end-min_loyal_period+1):end])
                if any(buying_last_k .== min_loyal_period)
                    loyal_brand = buying_last_k .== min_loyal_period
                    buyer.loyalty[loyal_brand] .= true
                end
            end
        end
    end

    return customers

end

function simulate(max_iter::Int64, num_firms::Int64, num_customers::Int64, a, σ_v0, σ_ϵ, σ_α, prices, advertising, max_stock_period, wealth, network_type, num_links, pref_attachment_links, dist, accessibility_step, λ_ind, λ_wom, λ_ad, method, risk, buyer_memory, min_loyal_period, loyalty_buying, only_loyalty_wom)
    """
    Function simulating the process.

    customers - vector of customers;
    firms - vector of firms;
    network - network connecting customers
    accessibility_step - def. neighbourhood, 1 - only direct neighbours, 2 - up to direct neighbours of my neighbours etc.
    λ_ind - learning rate, individual experience
    λ_wom - learning rate, neighbours experience
    λ_ad - learning rate, advertising exposure
    dist - distribution of sampling, uniform or trimmed normal

    """

    firms = create_firms(num_firms, a, σ_v0, prices, advertising, σ_ϵ, σ_α)
    customers = create_customers(num_customers, firms, a, σ_v0, max_stock_period, wealth, risk)
    network = create_network(network_type; num_customers = num_customers, num_links = num_links, pref_attachment_links = pref_attachment_links)

    sales = []
    learning = []
    uncertainty = []
    surpluses = []
    loyalty = []

    push!(uncertainty, reduce(+, getfield.(customers, :quality_uncertainty)))

    for iter in 1:max_iter
        customers = calculate_price_and_traded_units(customers, firms; max_stock_period = max_stock_period, iter = iter, method = method, loyalty_buying)
        customers = exchange_goods(customers, firms; dist = dist)
        customers = receive_advertising(customers, firms; intensity = ad_intensity, dist = dist, iter = iter)
        customers, learning_iter = update_quality_expectation_BAYESIAN(customers, firms, network; accessibility_step = accessibility_step, λ_ind = λ_ind, λ_wom = λ_wom, λ_ad = λ_ad, dist=dist, iter=iter, buyer_memory = buyer_memory, only_loyalty_wom = only_loyalty_wom)
        # customers = calculate_loyalty(customers, min_loyal_period, iter)
        push!(sales, reduce(+, getfield.(customers, :unit_bought)))
        push!(learning, learning_iter)
        push!(surpluses, reduce(+, getfield.(customers, :surplus)))
        push!(uncertainty, reduce(+, getfield.(customers, :quality_uncertainty)))
        push!(loyalty, reduce(+, getfield.(customers, :loyalty)))
    end

    return sales, learning, uncertainty, surpluses, (customers, firms)# , loyalty
end


function simulate_aggregated_results(max_iter::Int64, num_firms::Int64, num_customers::Int64, a, σ_v0, σ_ϵ, σ_α, prices, advertising, max_stock_period, wealth, network_type, num_links, pref_attachment_links, dist, accessibility_step, λ_ind, λ_wom, λ_ad, method, risk, buyer_memory, min_loyal_period, loyalty_buying, only_loyalty_wom)

    sales, learning, uncertainty, surpluses, cf = simulate(max_iter, num_firms, num_customers, a, σ_v0, σ_ϵ, σ_α, prices, advertising, max_stock_period, wealth, network_type, num_links, pref_attachment_links, dist, accessibility_step, λ_ind, λ_wom, λ_ad, method, risk, buyer_memory, min_loyal_period, loyalty_buying, only_loyalty_wom)

    sales_aggr = reduce(+, sales)

    return sales_aggr

end

#### SANDBOX ####

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
# tu zaczyna się simulate

SALES, LEARNING, UNCERTIANTY, SURPLUSES, CF = simulate(MAX_ITER, NUM_FIRMS, NUM_CUSTOMERS, A, Σ_V0, Σ_Ε, Σ_Α, PRICES, ADVERTISING, MAX_STOCK_PERIOD, WEALTH, NETWORK_TYPE, NUM_LINKS, PREF_ATTACHMENT_LINKS, DIST, ACCESSIBILITY_STEP, Λ_IND, Λ_WOM, Λ_AD, METHOD, RISK, BUYER_MEMORY, MIN_LOYAL_PERIOD, LOYALTY_BUYING, ONLY_LOYALTY_WOM)

@time simulate_aggregated_results(MAX_ITER, NUM_FIRMS, NUM_CUSTOMERS, A, Σ_V0, Σ_Ε, Σ_Α, PRICES, ADVERTISING, MAX_STOCK_PERIOD, WEALTH, NETWORK_TYPE, NUM_LINKS, PREF_ATTACHMENT_LINKS, DIST, ACCESSIBILITY_STEP, Λ_IND, Λ_WOM, Λ_AD, METHOD, RISK, BUYER_MEMORY, MIN_LOYAL_PERIOD, LOYALTY_BUYING, ONLY_LOYALTY_WOM)

groupedbar(transpose(hcat(1*CF[1][200].history_experience...)),bar_width=0.7, linecolor=nothing,
        bar_position = :stack)

CF[1][1].quality_expectation
CF[1][1].quality_uncertainty

plot(getindex.(LOYALTY, 1))
plot!(getindex.(LOYALTY, 2))
plot!(getindex.(LOYALTY, 3))

plot(reduce(+, [getindex.(getindex.(LEARNING, cc), 1) for cc in 1:NUM_CUSTOMERS]) ./ NUM_CUSTOMERS)
plot!(reduce(+, [getindex.(getindex.(LEARNING, cc), 2) for cc in 1:NUM_CUSTOMERS]) ./ NUM_CUSTOMERS)
plot!(reduce(+, [getindex.(getindex.(LEARNING, cc), 3) for cc in 1:NUM_CUSTOMERS]) ./ NUM_CUSTOMERS)

plot(getindex.(UNCERTIANTY, 1) ./ NUM_CUSTOMERS, ylim = (0,1))
plot!(getindex.(UNCERTIANTY, 2) ./ NUM_CUSTOMERS)
plot!(getindex.(UNCERTIANTY, 3) ./ NUM_CUSTOMERS)

plot(getindex.(SURPLUSES, 1))
plot!(getindex.(SURPLUSES, 2))
plot!(getindex.(SURPLUSES, 3))

groupedbar(transpose(hcat(SALES...)), bar_width=0.7, linecolor=nothing,
        bar_position = :stack)



@time FIRMS = create_firms(NUM_FIRMS, A, Σ_V0, PRICES, ADVERTISING,Σ_Ε,Σ_Α)
@time CUSTOMERS = create_customers(NUM_CUSTOMERS, FIRMS, A, Σ_V0, MAX_STOCK_PERIOD, WEALTH, RISK)
@time NETWORK = create_network(NETWORK_TYPE; num_customers = NUM_CUSTOMERS, num_links = NUM_LINKS, pref_attachment_links = PREF_ATTACHMENT_LINKS)

ITER = 1

@time CUSTOMERS = calculate_price_and_traded_units(CUSTOMERS, FIRMS; max_stock_period = MAX_STOCK_PERIOD, iter = ITER, method = METHOD, loyalty_buying = LOYALTY_BUYING)

@time CUSTOMERS = exchange_goods(CUSTOMERS, FIRMS; dist = DIST)
@time CUSTOMERS = receive_advertising(CUSTOMERS, FIRMS; intensity = AD_INTENSITY, dist = DIST, iter = ITER)
@time CUSTOMERS, LEARNING_ITER = update_quality_expectation_BAYESIAN(CUSTOMERS, FIRMS, NETWORK; accessibility_step = ACCESSIBILITY_STEP, λ_ind = Λ_IND, λ_wom = Λ_WOM, λ_ad = Λ_AD, dist=DIST, iter=ITER, buyer_memory = BUYER_MEMORY, only_loyalty_wom = ONLY_LOYALTY_WOM)

0.009*600