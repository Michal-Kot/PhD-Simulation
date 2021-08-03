using LightGraphs
using GraphPlot
using Distributions
using Plots

mutable struct seller
    id::Int64
    marginal_cost::Int64
    surplus::Float64
end

mutable struct buyer
    id::Int64
    std_reservation_price::Int64
    quality_expectation::Float64
    reservation_price::Float64
    unit_bought::Bool
    quality_of_unit_bought::Float64
    surplus::Float64
    received_ad::Bool
    quality_of_unit_ad::Float64
end



# FUNCTIONS

function create_network(type::String; num_buyers::Int64, num_links::Int64=0, pref_attachment_links::Int64=0)
    if type == "random"
        return SimpleGraph(num_buyers, num_links)
    elseif type == "preferential_attachment"
        return barabasi_albert(num_buyers, pref_attachment_links)
    end
end

function create_sellers(num_sellers::Int64)::Vector{seller}
    sellers_vector = []
    for s in 1:num_sellers
        new_seller = seller(s, s, 0.0)
        push!(sellers_vector, new_seller)
    end
    return sellers_vector
end

function create_buyers(num_buyers::Int64)::Vector{buyer}
    buyers_vector = []
    for b in 1:num_buyers
        new_buyer = buyer(b, b, 1.0, b * 1.0, false, 0.0, 0.0, false, 0.0)
        push!(buyers_vector, new_buyer)
    end
    return buyers_vector
end

function sample_quality(dist::String; quality_variance::Float64=0)
    if dist == "uniform"
        qv = minimum([quality_variance, 1/3])
        qv = sqrt(12 * qv)
        return 1 - qv/2 + rand(Uniform(0, qv))
    elseif dist == "trimmed_normal"
        sampled_quality = rand(Normal(1, quality_variance))
        sampled_quality = minimum([sampled_quality,2])
        sampled_quality = maximum([sampled_quality,0])
        return sampled_quality
    elseif dist == "exponential"
        return rand(Exponential(1))
    end
end

function create_supply(sellers::Vector{seller})
    sell_order = sortperm(getfield.(sellers, :marginal_cost))
    ordered_sellers = sellers[sell_order]
    return ordered_sellers
end

function create_demand(buyers::Vector{buyer})
    buy_order = sortperm(getfield.(buyers, :reservation_price), rev = true)
    ordered_buyers = buyers[buy_order]
    return ordered_buyers
end

function calculate_price_and_traded_units(buyers::Vector{buyer}, sellers::Vector{seller})

    sellers_in_round = Vector{Int64}()
    buyers_in_round = Vector{Int64}()
    traded_units = 1
    min_price = 0
    max_price = 0
    num_buyers = length(buyers)
    num_sellers = length(sellers)

    setfield!.(buyers, :unit_bought, false)
    setfield!.(buyers, :received_ad, false)

    while (traded_units < num_buyers) & (traded_units < num_sellers) & (buyers[traded_units].reservation_price >= sellers[traded_units].marginal_cost)
        push!(sellers_in_round, sellers[traded_units].id)
        push!(buyers_in_round, buyers[traded_units].id)
        traded_units += 1
    end

    if traded_units > 0
        traded_units -= 1
    end

    if traded_units > 0
        if num_sellers == traded_units
            min_price = buyers[traded_units].reservation_price
            else
            min_price = minimum([buyers[traded_units].reservation_price, sellers[traded_units+1].marginal_cost])
        end
        if num_buyers == traded_units
            max_price = sellers[traded_units].marginal_cost
            else
            max_price = maximum([buyers[traded_units+1].reservation_price, sellers[traded_units].marginal_cost])
        end
    end

    price = 1/2 * (min_price + max_price)

    return (sellers_in_round = sellers_in_round, buyers_in_round = buyers_in_round, traded_units = traded_units, price = price)
end

function exchange_goods(buyers::Vector{buyer}, sellers::Vector{seller}, market; dist, quality_variance)

    for unit in 1:market.traded_units
        prod_qual = sample_quality(dist; quality_variance = quality_variance)
        buyer_id = market.buyers_in_round[unit]
        buyer_number = argmax(getfield.(buyers, :id) .== buyer_id)
        setfield!(buyers[buyer_number], :unit_bought, true)
        setfield!(buyers[buyer_number], :quality_of_unit_bought, prod_qual)
        setfield!(buyers[buyer_number], :surplus, prod_qual * buyers[buyer_number].std_reservation_price - market.price)

        seller_id = market.sellers_in_round[unit]
        seller_number = argmax(getfield.(sellers, :id) .== seller_id)
        setfield!(sellers[seller_number], :surplus, market.price - sellers[seller_number].marginal_cost)

    end

    return buyers, sellers

end

function update_quality_expectation(method::String, buyers::Vector{buyer}, network; accessibility_step::Int64=1, λ_ind::Float64=0.0, λ_neigh::Float64=0.0, λ_ad::Float64=0.0)
    if method == "IZQUIERDO"
        return update_quality_expectation_IZQ(buyers, network; accessibility_step, λ_ind, λ_neigh, λ_ad)
    elseif method == "BAYES"
        return 0
    end
end

function update_quality_expectation_IZQ(buyers::Vector{buyer}, network; accessibility_step::Int64=1, λ_ind::Float64=0.0, λ_neigh::Float64=0.0, λ_ad::Float64=0.0)

    for i in 1:length(buyers)

        my_prod_qual = 0
        social_prod_qual = Vector{Float64}()
        my_ad_qual = 0

        if buyers[i].unit_bought
            my_prod_qual = buyers[i].quality_of_unit_bought - buyers[i].quality_expectation
        end

        my_id = getfield(buyers[i], :id)
        my_neighbours_distance = gdistances(network, my_id)
        my_neighbours_ids = collect(1:length(buyers))[(my_neighbours_distance .<= accessibility_step) .& (my_neighbours_distance .> 0)]
        my_neighbours_positions = .!isnothing.(indexin(getfield.(buyers, :id), my_neighbours_ids))
        my_neighbours_positions = findall(x->x==maximum(my_neighbours_positions), my_neighbours_positions)

        neigh_prod_quality = 0
        neigh_prod_buy = 0
        mean_neigh_prod_qual = 0

        for neigh in my_neighbours_positions
            if buyers[neigh].unit_bought
                neigh_prod_quality += buyers[neigh].quality_of_unit_bought
                neigh_prod_buy += 1
            end
        end

        if neigh_prod_buy > 0
            mean_neigh_prod_qual = neigh_prod_quality / neigh_prod_buy - buyers[i].quality_expectation
        end

        if buyers[i].received_ad
            my_ad_qual = buyers[i].quality_of_unit_ad - buyers[i].quality_expectation
        end

        new_quality_expectation = buyers[i].quality_expectation + λ_ind * my_prod_qual + λ_neigh * mean_neigh_prod_qual + λ_ad * my_ad_qual

        setfield!(buyers[i], :quality_expectation, new_quality_expectation)
        setfield!(buyers[i], :reservation_price, new_quality_expectation * getfield(buyers[i], :std_reservation_price))

    end

    return buyers

end

function receive_advertising(buyers::Vector{buyer}; intensity::Float64, dist::String, quality_variance::Float64)

    sampled_ad_receivers = sample(1:length(buyers), Int(floor(intensity * length(buyers))),replace=false)

    #println(sampled_ad_receivers)

    for i in sampled_ad_receivers

        setfield!(buyers[i], :received_ad, true)
        setfield!(buyers[i], :quality_of_unit_ad, sample_quality(dist; quality_variance = quality_variance))

    end

    return buyers

end


function simulate(max_iter::Int64, num_sellers::Int64, num_buyers::Int64; network_type::String, num_links::Int64=0, pref_attachment_links::Int64=0, dist::String, quality_variance_exp::Float64, quality_variance_ad::Float64, accessibility_step::Int64=1, λ_ind::Float64, λ_neigh::Float64, ad_intensity::Float64=0, λ_ad::Float64, update_method::String="IZQUIERDO")

    """SETUP"""

    sellers = create_sellers(num_sellers)
    buyers = create_buyers(num_buyers)
    network = create_network(network_type; num_buyers = num_buyers, num_links = num_links, pref_attachment_links = pref_attachment_links)

    result_price = Vector{Float64}()
    result_traded_units = Vector{Float64}()
    result_average_expected_quality = Vector{Float64}()

    result_sellers_surplus = Vector{Float64}()
    result_buyers_surplus = Vector{Float64}()

    result_demand = []
    result_supply = (price = collect(1:(1+maximum(getfield.(sellers, :marginal_cost)))), volume = [sum(getfield.(sellers, :marginal_cost) .<= x) for x in 1:(1+maximum(getfield.(sellers, :marginal_cost)))])

    """GO"""

    for iteration in 1:max_iter

        sellers = create_supply(sellers)
        buyers = create_demand(buyers)

        market = calculate_price_and_traded_units(buyers, sellers)

        push!(result_demand, (price = collect(1:(1+maximum(getfield.(buyers, :reservation_price)))), volume = [sum(getfield.(buyers, :reservation_price) .>= x) for x in 1:(1+maximum(getfield.(buyers, :reservation_price)))]))

        push!(result_price, market.price)
        push!(result_traded_units, market.traded_units)

        buyers, sellers = exchange_goods(buyers, sellers, market; dist = dist, quality_variance = quality_variance_exp)

        push!(result_buyers_surplus, sum(getfield.(buyers, :surplus)))
        push!(result_sellers_surplus, sum(getfield.(sellers, :surplus)))

        buyers = receive_advertising(buyers; intensity = ad_intensity, dist = dist, quality_variance = quality_variance_ad)

        buyers = update_quality_expectation(update_method, buyers, network; accessibility_step = accessibility_step, λ_ind = λ_ind, λ_neigh = λ_neigh, λ_ad = λ_ad)

        push!(result_average_expected_quality, mean(getfield.(buyers, :quality_expectation)))

    end

    return (price = result_price, traded_units = result_traded_units, expected_quality = result_average_expected_quality, buyers_surplus = result_buyers_surplus, sellers_surplus = result_sellers_surplus, demand = result_demand, supply = result_supply)

end

# VARIABLES CONTROLLING THE SIMULATION

MAX_ITER = 1000 # number of simulation iterations

NUM_BUYERS = 100 # number of buyers
NUM_SELLERS = 100 # number of sellers

NETWORK_TYPE = "preferential_attachment" # type of network between agents
NUM_LINKS = 200 # number of links, for random graphs only
PREF_ATTACHMENT_LINKS = 6 # number of neighbours per agent, for barabasi-albert only

DIST = "uniform" # distribution of quality
QUALITY_VARIANCE_EXP = 0.2 # variance of quality, when consuming

QUALITY_VARIANCE_AD = 0.6 # variance of quality, when ad
ACCESSIBILITY_STEP = 1 # distance from agent to agent, defining neighbourhood
Λ_IND = 0.8 # individual learning
Λ_NEIGH = 0.2 # social learning
Λ_AD = 0.5

AD_INTENSITY = 0.6

# SIMULATION

@time RESULTS = simulate(MAX_ITER, NUM_SELLERS, NUM_BUYERS; network_type=NETWORK_TYPE, pref_attachment_links = PREF_ATTACHMENT_LINKS, dist = DIST, quality_variance_exp =  QUALITY_VARIANCE_EXP, quality_variance_ad = QUALITY_VARIANCE_AD, accessibility_step = ACCESSIBILITY_STEP, λ_ind = Λ_IND, λ_neigh = Λ_NEIGH, ad_intensity = AD_INTENSITY, λ_ad = Λ_AD)

# RESULTS

plot(RESULTS.price, xlabel = "Trading sessions", ylabel = "Price", legend = nothing)
plot!(RESULTS.traded_units, xlabel = "Trading sessions", ylabel = "Volume", legend = nothing)
plot(RESULTS.expected_quality, xlabel = "Trading sessions", ylabel = "Average expected quality", legend = nothing)

plot(RESULTS.buyers_surplus, xlabel = "Trading sessions", ylabel = "Surplus", label = "Buyers")
plot!(RESULTS.sellers_surplus, xlabel = "Trading sessions", label = "Sellers")
plot!(RESULTS.buyers_surplus + RESULTS.sellers_surplus, xlabel = "Trading sessions", label = "Total")

plot(RESULTS.supply.volume, RESULTS.supply.price, xlabel = "Volume", ylabel = "Price", label = "Supply")
plot!(RESULTS.demand[1].volume, RESULTS.demand[1].price, label = "Demand (t=0)")
plot!(RESULTS.demand[10].volume, RESULTS.demand[10].price, label = "Demand (t=10)")
plot!(RESULTS.demand[100].volume, RESULTS.demand[100].price, label = "Demand (t=100)")
plot!(RESULTS.demand[1000].volume, RESULTS.demand[1000].price, label = "Demand (t=1000)")

#### SANDBOX ####

num_sellers = NUM_SELLERS
num_buyers = NUM_BUYERS
network_type = NETWORK_TYPE = "preferential_attachment" # type of network between agents
num_links = NUM_LINKS = 200 # number of links, for random graphs only
pref_attachment_links = PREF_ATTACHMENT_LINKS = 6 # number of neighbours per agent, for barabasi-albert only

dist = DIST = "uniform" # distribution of quality
quality_variance_exp = QUALITY_VARIANCE_EXP = 0.2 # variance of quality, when consuming

quality_variance_ad = QUALITY_VARIANCE_AD = 0.6 # variance of quality, when ad
accessibility_step = ACCESSIBILITY_STEP = 1 # distance from agent to agent, defining neighbourhood
Λ_IND = 0.8 # individual learning
Λ_NEIGH = 0.2 # social learning
Λ_AD = 0.5

ad_intensity = AD_INTENSITY = 0.6

i = 1

my_prod_qual = 0
social_prod_qual = Vector{Float64}()
my_ad_qual = 0

if buyers[i].unit_bought
    my_prod_qual = buyers[i].quality_of_unit_bought - buyers[i].quality_expectation
end

my_id = getfield(buyers[i], :id)
my_neighbours_distance = gdistances(network, my_id)
my_neighbours_ids = collect(1:length(buyers))[(my_neighbours_distance .<= accessibility_step) .& (my_neighbours_distance .> 0)]
my_neighbours_positions = .!isnothing.(indexin(getfield.(buyers, :id), my_neighbours_ids))
my_neighbours_positions = findall(x->x==maximum(my_neighbours_positions), my_neighbours_positions)

neigh_prod_quality = 0
neigh_prod_buy = 0
mean_neigh_prod_qual = 0

for neigh in my_neighbours_positions
    if buyers[neigh].unit_bought
        neigh_prod_quality += buyers[neigh].quality_of_unit_bought
        neigh_prod_buy += 1
    end
end

if neigh_prod_buy > 0
    mean_neigh_prod_qual = neigh_prod_quality / neigh_prod_buy - buyers[i].quality_expectation
end

if buyers[i].received_ad
    my_ad_qual = buyers[i].quality_of_unit_ad - buyers[i].quality_expectation
end

quality_variance_exp

new_quality_expectation = buyers[i].quality_expectation + λ_ind * my_prod_qual + λ_neigh * mean_neigh_prod_qual + λ_ad * my_ad_qual

setfield!(buyers[i], :quality_expectation, new_quality_expectation)
setfield!(buyers[i], :reservation_price, new_quality_expectation * getfield(buyers[i], :std_reservation_price))
