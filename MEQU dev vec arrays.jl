using LightGraphs: issymmetric
using LightGraphs
using GraphPlot
using Distributions
using Plots
using StatsPlots
using StatsBase
using LinearAlgebra

# SANDBOX


function create_f_price(J, T, prices)
    f_price = zeros(Float64, J, T)
    for j in 1:J
        f_price[j,:] = prices[j]
    end
    return f_price
end

function create_f_advertising(J, T, advertising)
    f_advertising = zeros(Float64, J, T)
    for j in 1:J
        f_advertising[j,:] = advertising[j]
    end
    return f_advertising
end

function create_f_quality(A)
    f_quality = A
    return f_quality
end

function create_f_quality_variance(σ_v0)
    f_quality_variance = σ_v0
    return f_quality_variance
end


function create_f_quality_variance_experience(σ_ϵ)
    f_quality_variance_experience = σ_ϵ
    return f_quality_variance_experience
end


function create_f_quality_variance_advertising(σ_α)
    f_quality_variance_advertising = σ_α
    return f_quality_variance_advertising
end

function create_c_std_reservation_price(I, wealth)
    c_std_reservation_price = rand(TriangularDist(0,100,wealth), I)
    return c_std_reservation_price
end

function create_c_quality_expectation(I,J,T,A)
    c_quality_expectation = zeros(Float64, I, J, T)
    for i in 1:I
        c_quality_expectation[i,:,1] .= A
    end
    return c_quality_expectation
end

function create_c_quality_uncertainty(I,J,T,σ_v0)
    c_quality_uncertainty = zeros(Float64, I, J ,T)
    for i in 1:I
        c_quality_uncertainty[i,:,1] .= σ_v0
    end
    return c_quality_uncertainty
end

function create_c_stock(I,T,max_stock_period, c_std_reservation_price)
    c_stock = zeros(Int64, I, T)
    #c_stock[:,1] = sample(1:max_stock_period, I)
    c_stock[:,1] .= 1
    #c_stock[:,1] .= Int.(floor.(c_std_reservation_price ./ 100 * max_stock_period))
    return c_stock
end

function create_c_risk(I,risk)
    c_risk = rand(TriangularDist(0,100,risk), I)
    return c_risk
end

function inverse_gdistances(i; network)
    return gdistances(network, i)
end

function find_all_neighbours_id(neighbours_distances; accessibility_step)
    return findall((neighbours_distances .> 0) .& (neighbours_distances .<= accessibility_step))
end

function create_network(network_type::String, I::Int64, num_links::Int64=0, pref_attachment_links::Int64=0, accessibility_step::Int64=1)
    """
    Function creating network

    type - def. network constructor type, either random or preferential attachment;
    num_customers - def. problem size, number of agents => number of vertices;
    num_links - def. total number of links, for random network only;
    pref_attachment_links - def. number of each agent's connections, for preferential attachment network only.

    """

    g = SimpleGraph(I)

    if network_type == "random"
        g = SimpleGraph(num_customers, num_links)
    elseif network_type == "preferential_attachment"
        g = barabasi_albert(num_customers, pref_attachment_links)
    end

    neighbours_distances = inverse_gdistances.(1:I; network=g)

    ngh = find_all_neighbours_id.(neighbours_distances; accessibility_step)

    network = zeros(Bool, I, I)

    for i1 in 1:I
        for i2 in 1:I
            if i2 ∈ ngh[i1]
            network[i1,i2] = true
            end
        end
    end

    return network

end



# STOCKS

function use_stock(c_stock, iter)
    c_stock[:, iter] .= max.(c_stock[:, iter-1] .- 1, 0)
    return c_stock
end


# CONSUMER customer_choice


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

function customer_choice(I, J, c_stock, c_std_reservation_price, c_quality_expectation, c_quality_uncertainty, c_risk, c_quality_of_unit_bought, f_price, f_quality, f_quality_variance_experience, c_unit_bought, iter)

    c_willing_to_buy = c_stock[:,iter] .<= 0
    c_wtp = c_std_reservation_price .* c_quality_expectation[:,:,iter-1]
    c_utility = c_wtp .- c_risk .* c_quality_uncertainty[:,:,iter-1]
    c_surplus = c_wtp .- repeat(transpose(f_price[:, iter]), inner = (I,1))

    for i in 1:I
        if c_willing_to_buy[i]
            if any(c_surplus[i,:] .> 0)
                b,u,s = choose_best(1:J, c_utility[i,:], c_surplus[i,:])
                c_unit_bought[i, b, iter] = true
                c_quality_of_unit_bought[i,b,iter] = f_quality[b] + rand(Normal(0, f_quality_variance_experience[b]))
                c_stock[i,iter] = sample(1:max_stock_period)
            end
        end
    end

    return c_stock, c_unit_bought, c_quality_of_unit_bought

end

function receive_ad(I, c_received_ad, c_quality_of_received_ad, f_advertising, f_quality, f_quality_variance_advertising, iter)
    for i in 1:I
        c_received_ad[i,:,iter] = BitVector(rand.(Binomial.(1, f_advertising[:,iter])))
        c_quality_of_received_ad[i,:,iter] = c_received_ad[i,:,iter] .* (f_quality .+ rand.(Normal.(0, f_quality_variance_advertising)))
    end
    return c_received_ad,  c_quality_of_received_ad
end

function calculate_memory(iter, buyer_memory)

    k = buyer_memory

    if iter < buyer_memory
        k = iter
    end

    return k

end

function bayesian_updating(I, J, T, c_quality_of_unit_bought, c_quality_uncertainty, c_quality_expectation, c_unit_bought, c_neighbours_bought, c_received_ad, c_quality_of_received_ad, f_quality_variance, f_quality_variance_experience, f_quality_variance_advertising, network, iter, buyer_memory, λ_ind, λ_wom, λ_ad)

    f_base_variance2 = f_quality_variance .^ 2
    f_quality_variance_experience2 = f_quality_variance_experience.^2
    f_quality_variance_advertising2 = f_quality_variance_advertising.^2
    f_base_uncertainty = 1 ./ f_base_variance2

    c_consumption_surprise = (c_quality_of_unit_bought[:,:,iter] .- c_quality_expectation[:,:,iter-1]) .* c_unit_bought[:,:,iter]

    c_quality_of_neighbours_bought = zeros(Float64, I, J)

    for i in 1:I
        c_quality_of_neighbours_bought[i,:] = sum(c_quality_of_unit_bought[network[:,i],:,iter], dims=1)
        c_neighbours_bought[i,:,iter] = sum(c_unit_bought[network[:,i],:,iter], dims=1)
    end

    c_quality_of_neighbours_bought_avg = c_quality_of_neighbours_bought ./ c_neighbours_bought[:,:,iter]
    c_quality_of_neighbours_bought_avg[isnan.(c_quality_of_neighbours_bought_avg)] .= 0

    c_wom_surprise = (c_quality_of_neighbours_bought_avg .- c_quality_expectation[:,:,iter-1]) .* sign.(c_neighbours_bought[:,:,iter])

    c_ad_surprise = (c_quality_of_received_ad[:,:,iter] .- c_quality_expectation[:,:,iter-1]) .* c_received_ad[:,:,iter]

    backward_memory = calculate_memory(iter-1, buyer_memory)

    c_history_experience = sum(c_unit_bought[:,:,(iter-backward_memory+1):iter], dims=3)
    c_history_wom = sum(sign.(c_neighbours_bought[:,:,(iter-backward_memory+1):iter]), dims=3)
    c_history_advertising = sum(c_received_ad[:,:,(iter-backward_memory+1):iter], dims=3)

    c_uncertainty_experience = λ_ind * c_history_experience ./ repeat(transpose(f_quality_variance_experience2), inner=(I,1))
    c_uncertainty_wom = λ_wom * c_history_wom ./ repeat(transpose(f_quality_variance_experience2), inner=(I,1))
    c_uncertainty_advertising = λ_ad * c_history_advertising ./ repeat(transpose(f_quality_variance_advertising2), inner=(I,1))

    c_quality_uncertainty[:,:,iter] = 1 ./ (repeat(transpose(f_base_uncertainty), inner = (I,1)) .+ c_uncertainty_experience .+ c_uncertainty_wom .+ c_uncertainty_advertising)

    c_kalman_experience = c_quality_uncertainty[:,:,iter] ./ (c_quality_uncertainty[:,:,iter] .+ repeat(transpose(f_quality_variance_experience2), inner = (I,1)))
    c_kalman_wom = c_quality_uncertainty[:,:,iter] ./ (c_quality_uncertainty[:,:,iter] .+ repeat(transpose(f_quality_variance_experience2), inner = (I,1)))
    c_kalman_advertising = c_quality_uncertainty[:,:,iter] ./ (c_quality_uncertainty[:,:,iter] .+ repeat(transpose(f_quality_variance_advertising2), inner = (I,1)))

    c_learning_experience = c_unit_bought[:,:,iter] .* c_kalman_experience .* c_consumption_surprise
    c_learning_wom = sign.(c_neighbours_bought[:,:,iter]) .* c_kalman_wom .* c_wom_surprise
    c_learning_advertising = c_received_ad[:,:,iter] .* c_kalman_advertising .* c_ad_surprise

    c_quality_expectation[:,:,iter] = c_quality_expectation[:,:,iter-1] .+ c_learning_experience .+ c_learning_wom .+ c_learning_advertising

    return c_quality_uncertainty, c_quality_expectation

end
# simulate
iter=2

function simulate(num_customers, num_firms, max_iter, prices, advertising, a, σ_v0, σ_ϵ, σ_α, wealth, max_stock_period, risk, network_type, num_links, pref_attachment_links, accessibility_step, λ_ind, λ_wom, λ_ad, buyer_memory)

    a_vec = fill(a, num_firms)
    σ_v0_vec = fill(σ_v0, num_firms)
    σ_ϵ_vec = fill(σ_ϵ, num_firms)
    σ_α_vec = fill(σ_α, num_firms)

    I = num_customers
    J = num_firms
    T = max_iter

    c_stock = zeros(Int64, I, T)
    c_risk = zeros(Float64, I, T)

    f_price = create_f_price(J, T, prices)
    f_advertising = create_f_price(J, T, advertising)
    f_quality = create_f_quality(a_vec)
    f_quality_variance = create_f_quality_variance(σ_v0_vec)
    f_quality_variance_experience = create_f_quality_variance_experience(σ_ϵ_vec)
    f_quality_variance_advertising = create_f_quality_variance_advertising(σ_α_vec)

    c_std_reservation_price = create_c_std_reservation_price(I, wealth)
    c_quality_expectation = create_c_quality_expectation(I,J,T,a_vec)
    c_quality_uncertainty = create_c_quality_uncertainty(I,J,T,σ_v0_vec)
    c_unit_bought = falses(I, J, T)
    c_quality_of_unit_bought = zeros(Float64, I, J, T)
    c_received_ad = falses(I, J, T)
    c_quality_of_received_ad = zeros(Float64, I, J ,T)
    c_stock = create_c_stock(I,T,max_stock_period,c_std_reservation_price)
    c_risk = create_c_risk(I, risk)

    c_neighbours_bought = zeros(Int64, I, J, T)

    network = create_network(network_type, I, 1, pref_attachment_links, accessibility_step)

    # loop

    for iter in 2:T

        c_stock = use_stock(c_stock, iter)

        c_stock, c_unit_bought, c_quality_of_unit_bought = customer_choice(I, J, c_stock, c_std_reservation_price, c_quality_expectation, c_quality_uncertainty, c_risk, c_quality_of_unit_bought, f_price, f_quality, f_quality_variance_experience, c_unit_bought, iter)

        c_received_ad, c_quality_of_received_ad = receive_ad(I, c_received_ad, c_quality_of_received_ad, f_advertising, f_quality, f_quality_variance_advertising, iter)

        c_quality_uncertainty, c_quality_expectation = bayesian_updating(I, J, T, c_quality_of_unit_bought, c_quality_uncertainty, c_quality_expectation, c_unit_bought, c_neighbours_bought, c_received_ad, c_quality_of_received_ad, f_quality_variance, f_quality_variance_experience, f_quality_variance_advertising, network, iter, buyer_memory, λ_ind, λ_wom, λ_ad)

    end

    return c_quality_uncertainty, c_quality_expectation, c_unit_bought, c_received_ad, c_neighbours_bought, c_stock

end

# dopisać simulate na range parametrów





@time res_quality_uncertainty, res_quality_expectation, res_unit_bought, res_received_ad, res_neighbours_bought, res_stock = simulate(num_customers, num_firms, max_iter, prices, advertising, a, σ_v0, σ_ϵ, σ_α, wealth, max_stock_period, risk, network_type, 1, pref_attachment_links, accessibility_step, λ_ind, λ_wom, λ_ad, buyer_memory)

buying = sum(res_unit_bought, dims=1)
buying = dropdims(buying, dims=1)
buying = buying[:,3:end]

groupedbar(transpose(buying), bar_position = :stack, lw=false, xlabel = "Time", ylabel = "Sales")

sortperm(buying[1,:])

sum(mean(buying, dims=2)) * 10

plot(buying[1,:])
plot!(buying[2,:])
plot!(buying[3,:])
plot!(buying[4,:])

scatter(vec(mean(res_stock, dims=1)), vec(sum(buying, dims=1)))

plot(buying[1,:])
plot!(prices[1], lw=2, color=:black)
plot!(advertising[1] .* 100, lw=2, color = :red)

expectation = mean(res_quality_expectation, dims=1)
expectation = dropdims(expectation, dims=1)

plot(expectation[1,:], xlabel = "Time", ylabel = "Expectation", label = "Brand 1")
plot!(expectation[2,:])
plot!(expectation[3,:])
plot!(expectation[4,:])
plot!(expectation[5,:])
plot!(expectation[6,:])

uncertainty = mean(res_quality_uncertainty, dims=1)
uncertainty = dropdims(uncertainty, dims=1)
uncertainty = uncertainty[:,2:end]

plot(uncertainty[1,:], xlabel = "Time", ylabel = "Uncertainty")
plot!(uncertainty[2,:])
plot!(uncertainty[3,:])
plot!(uncertainty[4,:])
plot!(uncertainty[5,:])
plot!(uncertainty[6,:])

plot(res_stock[1,:])
plot!(res_stock[2,:])
plot!(res_stock[3,:])
plot!(res_stock[104,:])

plot(transpose(sum(res_stock, dims=1)))

(res_unit_bought .+ res_received_ad .+ res_neighbours_bought)[:,:,2]
res_quality_uncertainty[:,:,1]

# RANGES

r_num_firms = [4]
r_num_customers = [1000]
r_network_type = ["preferential_attachment"]
r_pref_attachment_links = [2]
r_accessibility_step = [1]
r_λ_ind = [0.5]
r_λ_wom = [0.5] # social learning
r_λ_ad = [0.1]

r_a = [1.]
r_σ_v0 = [0.5]
r_σ_α = [0.5]
r_σ_ϵ = [0.5]

r_max_stock_period = [10]
r_wealth = [10.0]
r_risk = [8.0]
r_buyer_memory = [15000]

r_p = [50]
r_c = [30]
r_d = [0,2,4,6,8,10]

r_cpp = [1]
r_i = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
r_ai = [0, 0.01, 0.02, 0.03]

r_max_iter = [365]

reps = 10

params = vec(collect(Base.Iterators.product(r_num_firms, r_num_customers, r_network_type, r_pref_attachment_links, r_accessibility_step, r_λ_ind, r_λ_wom, r_λ_ad, r_a, r_σ_v0, r_σ_α, r_σ_ϵ, r_max_stock_period, r_wealth, r_risk, r_buyer_memory, r_p, r_c, r_d, r_cpp, r_i, r_ai, r_max_iter)))

function simulate_loop(reps, params)

    results = []
    k=1
    for prm in params
        println(round(k / length(params), digits = 3))
        for rp in 1:reps
            num_firms = prm[1]
            num_customers = prm[2]
            network_type = prm[3]
            num_links = 1
            pref_attachment_links = prm[4]
            accessibility_step = prm[5]
            λ_ind = prm[6]
            λ_wom = prm[7]
            λ_ad = prm[8]
            a = prm[9]
            σ_v0 = prm[10]
            σ_α = prm[11]
            σ_ϵ = prm[12]
            max_stock_period = prm[13]
            wealth = prm[14]
            risk = prm[15]
            buyer_memory = prm[16]
            p = prm[17]
            c = prm[18]
            d = prm[19]
            cpp = prm[20]
            i = prm[21]
            ai = prm[22]
            max_iter = prm[23]
            discount, prices, margin_real = create_price(p, c, d, max_iter, num_firms)
            investment, advertising = create_advertising(cpp, i, ai, max_iter, num_firms)
            push!(results, simulate(num_customers, num_firms, max_iter, prices, advertising, a, σ_v0, σ_ϵ, σ_α, wealth, max_stock_period, risk, network_type, num_links, pref_attachment_links, accessibility_step, λ_ind, λ_wom, λ_ad, buyer_memory))
        end
        k += 1
    end

    return results

end

RESULTS = simulate_loop(1, params)

buying = [dropdims(sum(rr[3],dims=1), dims = 1)[1,:] for rr in RESULTS]
margin = getindex.(params,17) .- getindex.(params, 18) .- getindex.(params, 19)

margin_total = sum.(buying .* margin)
discount = getindex.(params, 19)
advertising = getindex.(params, 20) .* getindex.(params, 23) .* (getindex.(params, 22))

sum_buying = sum.(buying)

sales = [mean(sum_buying[(discount .== x1) .& (advertising .== x2)]) for x1 in unique(discount), x2 in unique(advertising)]

using PyPlot
using Plots; pyplot()

p = surf(unique(advertising), unique(discount), sales)
show(p)
display(gcf())
plot(unique(advertising),unique(discount), sales,st=:surface,camera=(-30,30))

scatter(discount, margin_total)

# SANDBOX

num_firms = NUM_FIRMS = 4
num_customers = NUM_CUSTOMERS = 1000
network_type = NETWORK_TYPE = "preferential_attachment" # type of network between agents      
pref_attachment_links = PREF_ATTACHMENT_LINKS = 2 # number of neighbours per agent, for barabasi-albert only
accessibility_step = ACCESSIBILITY_STEP = 1 # distance from agent to agent, defining neighbourhood

λ_ind = Λ_IND = 0.5 # individual learning
λ_wom = Λ_WOM = 0.5 # social learning
λ_ad = Λ_AD = 0.1

#a = A = fill(1., NUM_FIRMS)
#σ_v0 = Σ_V0 = fill(0.5, 10)[1:NUM_FIRMS]
#σ_α = Σ_Α = fill(0.5, 10)[1:NUM_FIRMS]
#σ_ϵ = Σ_Ε = fill(0.5, 10)[1:NUM_FIRMS]

a = 1.
σ_v0 = 0.5
σ_ϵ = 0.5
σ_α = 0.5

max_iter = MAX_ITER = 365

function cdf_Tri(x,a,b,c)
    if x < a
        return 0
    elseif (x >= a) & (x < c)
        return (x-a)^2 / ((b-a)*(c-a))
    elseif (x >= c) & (x < b)
        return 1 - (((b-x)^2) / ((b-a)*(b-c)))
    else
        return 1
    end
end

1 - cdf_Tri(50,0,100,wealth)

#ADVERTISING = fill(fill(0.0, MAX_ITER), NUM_FIRMS)

max_stock_period = MAX_STOCK_PERIOD = 10
wealth = WEALTH = 10.0
risk = RISK = 8.0
buyer_memory = BUYER_MEMORY = 15000

p = 60
c = 30
d = 10

discount, prices, margin = create_price(p,c,d,max_iter, num_firms)

cpp = 1
its = 0.03
aits = 0.03

investment, advertising = create_advertising(cpp, its, aits, max_iter, num_firms)


#### firm 1 preparing its strategy ####

function create_price(p,c,d, max_iter, num_firms)

    prices = [[p for i in 1:max_iter] for j in 1:num_firms]
    cog = [[c for i in 1:max_iter] for j in 1:num_firms]
    margin = prices .- cog

    discount = [[0. for i in 1:max_iter] for j in 1:num_firms]
    discount[1] .= d 
    prices_trans = prices .- discount
    margin_real = prices .- discount .- cog

    return discount, prices_trans, margin_real

end

function create_advertising(cpp, its, aits, max_iter, num_firms)

    investment = cpp * its * max_iter + cpp * aits * max_iter
    advertising = [[its for i in 1:max_iter] for x in 1:num_firms]
    advertising[1] = [its + aits for i in 1:max_iter]

    return investment, advertising

end

cost = sum(discount[1][3:end] .* buying[1,:]) + investment + additional_investment
margin = sum(margin[1][3:end] .* buying[1,:])

roi = margin / cost - 1
