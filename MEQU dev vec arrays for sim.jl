using LightGraphs: issymmetric
using LightGraphs
using GraphPlot
using Distributions
using Plots
using StatsPlots
using StatsBase
using LinearAlgebra
using DataFrames, CSV, DataFramesMeta

# SANDBOX


function create_f_price(J::Int64, T::Int64, prices)

    """

    function for creation of 2-dim price plans
    it takes number of firms J, simulation length T and price plans (vector of vectors J x T) and reshapes them into 2-dim matrix of J rows and T columns

    """

    f_price = zeros(Float64, J, T)
    for j in 1:J
        f_price[j,:] = prices[j]
    end
    return f_price
end

function create_f_advertising(J::Int64, T::Int64, advertising)

    """

    function for creation of 2dim advertising plans
    it takes number of firms J, simulation length T and advertising plans (vector of vectors J x T) and reshapes them into 2dim matrix of J rows and T columns
        
    """

    f_advertising = zeros(Float64, J, T)
    for j in 1:J
        f_advertising[j,:] = advertising[j]
    end
    return f_advertising
end

function create_f_quality(A::Vector{Float64})::Vector{Float64}
    """

    function for creation of firms' quality vector
    it simply changes name of input to output :)
            
    """
    f_quality = A
    return f_quality
end

function create_f_quality_variance(σ2_v0)
    """

    function for creation of firms' base quality variability vector
    it simply changes name of input to output :)
            
    """
    f_quality_variance = σ2_v0
    return f_quality_variance
end


function create_f_quality_variance_experience(σ2_ϵ)
    """

    function for creation of firms' experience variability: used for simulation of product's unit quality purchased by consumer
    it simply changes name of input to output :)
            
    """
    f_quality_variance_experience = σ2_ϵ
    return f_quality_variance_experience
end


function create_f_quality_variance_advertising(σ2_α)
    f_quality_variance_advertising = σ2_α
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

function create_c_quality_uncertainty(I,J,T,σ2_v0)
    c_quality_uncertainty = zeros(Float64, I, J ,T)
    for i in 1:I
        c_quality_uncertainty[i,:,1] .= σ2_v0
    end
    return c_quality_uncertainty
end

function create_c_stock(I,T,max_stock_period, c_std_reservation_price)
    c_stock = zeros(Int64, I, T)
    c_stock[:,1] = sample(1:max_stock_period, I)
    #c_stock[:,1] .= 1
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
        g = SimpleGraph(I, num_links)
    elseif network_type == "preferential_attachment"
        g = barabasi_albert(I, pref_attachment_links)
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
    
                chosen_brand = sample(1:length(highest_surplus_choices))
    
                brands = brands[chosen_brand]
                utility = utility[chosen_brand]
                surplus = surplus[chosen_brand]
    
            end
    
        end
    
    end

    brands = brands[1]
    utility = utility[1]
    surplus = surplus[1]

    return brands, utility, surplus

end

function customer_choice(I, J, c_stock, c_std_reservation_price, c_quality_expectation, c_quality_uncertainty, c_risk, c_quality_of_unit_bought, f_price, f_quality, f_quality_variance_experience, c_unit_bought, iter, max_stock_period)

    c_willing_to_buy = c_stock[:,iter] .<= 0
    c_wtp = c_std_reservation_price .* c_quality_expectation[:,:,iter-1]
    c_utility = c_wtp .- c_risk .* c_quality_uncertainty[:,:,iter-1]
    c_surplus = c_wtp .- repeat(transpose(f_price[:, iter]), inner = (I,1))

    for i in 1:I
        if c_willing_to_buy[i]
            if any(c_surplus[i,:] .> 0)
                b,u,s = choose_best(1:J, c_utility[i,:], c_surplus[i,:])
                c_unit_bought[i, b, iter] = true
                c_quality_of_unit_bought[i,b,iter] = f_quality[b] + rand(Normal(0, sqrt(f_quality_variance_experience[b])))
                c_stock[i,iter] = sample(1:max_stock_period)
            end
        end
    end

    return c_stock, c_unit_bought, c_quality_of_unit_bought

end

function receive_ad(I, c_received_ad, c_quality_of_received_ad, f_advertising, f_quality, f_quality_variance_advertising, iter)
    for i in 1:I
        c_received_ad[i,:,iter] = BitVector(rand.(Binomial.(1, f_advertising[:,iter])))
        c_quality_of_received_ad[i,:,iter] = c_received_ad[i,:,iter] .* (f_quality .+ rand.(Normal.(0, sqrt.(f_quality_variance_advertising))))
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

    f_base_variance = f_quality_variance
    f_quality_variance_experience = f_quality_variance_experience
    f_quality_variance_advertising = f_quality_variance_advertising
    f_base_uncertainty = 1 ./ f_base_variance

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

    c_uncertainty_experience = λ_ind * c_history_experience ./ repeat(transpose(f_quality_variance_experience), inner=(I,1))
    c_uncertainty_wom = λ_wom * c_history_wom ./ repeat(transpose(f_quality_variance_experience), inner=(I,1))
    c_uncertainty_advertising = λ_ad * c_history_advertising ./ repeat(transpose(f_quality_variance_advertising), inner=(I,1))

    c_quality_uncertainty[:,:,iter] = 1 ./ (repeat(transpose(f_base_uncertainty), inner = (I,1)) .+ c_uncertainty_experience .+ c_uncertainty_wom .+ c_uncertainty_advertising)

    c_kalman_experience = c_quality_uncertainty[:,:,iter] ./ (c_quality_uncertainty[:,:,iter] .+ repeat(transpose(f_quality_variance_experience), inner = (I,1)))
    c_kalman_wom = c_quality_uncertainty[:,:,iter] ./ (c_quality_uncertainty[:,:,iter] .+ repeat(transpose(f_quality_variance_experience), inner = (I,1)))
    c_kalman_advertising = c_quality_uncertainty[:,:,iter] ./ (c_quality_uncertainty[:,:,iter] .+ repeat(transpose(f_quality_variance_advertising), inner = (I,1)))

    c_learning_experience = c_unit_bought[:,:,iter] .* c_kalman_experience .* c_consumption_surprise
    c_learning_wom = sign.(c_neighbours_bought[:,:,iter]) .* c_kalman_wom .* c_wom_surprise
    c_learning_advertising = c_received_ad[:,:,iter] .* c_kalman_advertising .* c_ad_surprise

    c_quality_expectation[:,:,iter] = c_quality_expectation[:,:,iter-1] .+ c_learning_experience .+ c_learning_wom .+ c_learning_advertising

    return c_quality_uncertainty, c_quality_expectation

end
# simulate

function simulate(num_customers, num_firms, max_iter, prices, advertising, a, σ2_v0, σ2_ϵ, σ2_α, wealth, max_stock_period, risk, network_type, num_links, pref_attachment_links, accessibility_step, λ_ind, λ_wom, λ_ad, buyer_memory)

    a_vec = fill(a, num_firms)
    σ2_v0_vec = fill(σ2_v0, num_firms)
    σ2_ϵ_vec = fill(σ2_ϵ, num_firms)
    σ2_α_vec = fill(σ2_α, num_firms)

    I = num_customers
    J = num_firms
    T = max_iter

    c_stock = zeros(Int64, I, T)
    c_risk = zeros(Float64, I, T)

    f_price = create_f_price(J, T, prices)
    f_advertising = create_f_price(J, T, advertising)
    f_quality = create_f_quality(a_vec)
    f_quality_variance = create_f_quality_variance(σ2_v0_vec)
    f_quality_variance_experience = create_f_quality_variance_experience(σ2_ϵ_vec)
    f_quality_variance_advertising = create_f_quality_variance_advertising(σ2_α_vec)

    c_std_reservation_price = create_c_std_reservation_price(I, wealth)
    c_quality_expectation = create_c_quality_expectation(I,J,T,a_vec)
    c_quality_uncertainty = create_c_quality_uncertainty(I,J,T,σ2_v0_vec)
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

        c_stock, c_unit_bought, c_quality_of_unit_bought = customer_choice(I, J, c_stock, c_std_reservation_price, c_quality_expectation, c_quality_uncertainty, c_risk, c_quality_of_unit_bought, f_price, f_quality, f_quality_variance_experience, c_unit_bought, iter, max_stock_period)

        c_received_ad, c_quality_of_received_ad = receive_ad(I, c_received_ad, c_quality_of_received_ad, f_advertising, f_quality, f_quality_variance_advertising, iter)

        c_quality_uncertainty, c_quality_expectation = bayesian_updating(I, J, T, c_quality_of_unit_bought, c_quality_uncertainty, c_quality_expectation, c_unit_bought, c_neighbours_bought, c_received_ad, c_quality_of_received_ad, f_quality_variance, f_quality_variance_experience, f_quality_variance_advertising, network, iter, buyer_memory, λ_ind, λ_wom, λ_ad)

    end

    return c_quality_uncertainty, c_quality_expectation, c_unit_bought, c_received_ad, c_neighbours_bought, c_stock

end

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

function simulate_loop(reps, parameters)

    results_total = Vector{Any}(nothing,length(parameters))

    Threads.@threads for itr in 1:length(parameters)

        println(itr)
    
        prm = parameters[itr]

        results_chunk = []

         for rp in 1:reps
            num_firms = prm[1]
            num_customers = prm[2]
            network_type = prm[3]
            num_links = prm[23]
            pref_attachment_links = prm[4]
            accessibility_step = prm[5]
            λ_ind = prm[6]
            λ_wom = prm[7]
            λ_ad = prm[8]
            a = prm[9]
            σ2_v0 = prm[10]
            σ2_α = prm[11]
            σ2_ϵ = prm[12]
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
            max_iter = prm[24]
            discount, prices, margin_real = create_price(p, c, d, max_iter, num_firms)
            investment, advertising = create_advertising(cpp, i, ai, max_iter, num_firms, num_customers)
            res_quality_uncertainty, res_quality_expectation, res_unit_bought, res_received_ad, res_neighbours_bought, res_stock = simulate(num_customers, num_firms, max_iter, prices, advertising, a, σ2_v0, σ2_ϵ, σ2_α, wealth, max_stock_period, risk, network_type, num_links, pref_attachment_links, accessibility_step, λ_ind, λ_wom, λ_ad, buyer_memory)
            buying = sum(res_unit_bought, dims=1)
            buying = dropdims(buying, dims=1)
            buying = buying[1,2:end]
            total_margin = sum(margin_real[1][2:end] .* buying)
            price_cost = sum(discount[1][2:end] .* buying)
            ad_cost = sum(investment)
            total_cost = price_cost + ad_cost
            roi = total_margin / total_cost

            market_size = sum(res_unit_bought[:,:,2:end])

            firm1_size = sum(buying)

            firm1_final_quality = mean(res_quality_expectation[:,1,end])

            firm1_final_uncertainty = mean(res_quality_uncertainty[:,1,end])


            #file_dir = "C:/Users/mihau/Documents/phd_results_loop.csv"
            #results_df = DataFrame(discount = d, intentisty = i, additional_intensity = ai, roi=roi, margin = total_margin, cost = total_cost)
            #CSV.write(file_dir, results_df, append=true, writeheader = (k==1), delim=';', decimal=',')
            push!(results_chunk, (roi = roi, margin = total_margin, cost_investment = total_cost, market_size = market_size, sales_quantity = firm1_size, final_quanlity_expectation = firm1_final_quality, final_uncertainty = firm1_final_uncertainty))
        end

        results_total[itr] = results_chunk
        
    end

    

    return results_total

end



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

function create_advertising(cpp::Float64, its::Float64, aits::Float64, max_iter::Int64, num_firms::Int64, num_customers::Int64)

    investment = (cpp * its * max_iter + cpp * aits * max_iter) * 100
    advertising = [[its for i in 1:max_iter] for x in 1:num_firms]
    advertising[1] = [its + aits for i in 1:max_iter]

    return investment, advertising

end

# dopisać simulate na range parametrów

""" Plan eksperymentów

- price elasticity vd. advertising - DONE
- price vs. advertising optimization - dobranie odpowiedniego cpp
- what if advertising 0, price discount 0 and no connections?

"""

# RANGES

r_num_firms = [4]
r_num_customers = [200]
r_network_type = ["random"]
r_num_links = [0,40,80,120,160,200]
r_pref_attachment_links = [1]
r_accessibility_step = [1]
r_λ_ind = [0.,1.]
r_λ_wom = [0.,1.] # social learning
r_λ_ad = [0.,1.]

r_a = [1.]
r_σ2_v0 = [0.5]
r_σ2_α = [0.5]
r_σ2_ϵ = [0.5]

r_max_stock_period = [20]
r_wealth = [50.]
r_risk = [50.]
r_buyer_memory = [15000]

r_p = [60]
r_c = [30]
r_d = [0.,2.5,5.,7.5,10.,12.5,15.,17.5,20.,22.5,25.,27.5,30.]

r_cpp = [1.]
r_i = [0.,0.025,0.050,0.075,0.10]
r_ai = [0.]

r_max_iter = [365]

reps = 10

params = vec(collect(Base.Iterators.product(r_num_firms, r_num_customers, r_network_type, r_pref_attachment_links, r_accessibility_step, r_λ_ind, r_λ_wom, r_λ_ad, r_a, r_σ2_v0, r_σ2_α, r_σ2_ϵ, r_max_stock_period, r_wealth, r_risk, r_buyer_memory, r_p, r_c, r_d, r_cpp, r_i, r_ai, r_num_links, r_max_iter)))

length(params)*reps

"""

Zbierane metryki

- liczba sprzedanych sztuk
- totalna liczba sprzedanych sztuk
- market share
- srednia finalna niepewnosc
- srednia finalna ocena jakosci


"""

@time RESULTS = simulate_loop(reps, params)

PARAM_DF = DataFrame(vcat(fill.(params, reps)...))
RES_DF = DataFrame(vcat(RESULTS...))

FINAL_DF = hcat(PARAM_DF, RES_DF)

CSV.write("C:/Users/mihau/Documents/phd_results_final_no_signals.csv", FINAL_DF, delim = ";", decimal = ',')

#### SANDBOX simulate


num_firms = NUM_FIRMS = 4
num_customers = NUM_CUSTOMERS = 1000
network_type = NETWORK_TYPE = "random" # type of network between agents      
pref_attachment_links = PREF_ATTACHMENT_LINKS = 2 # number of neighbours per agent, for barabasi-albert only
num_links = 1000
accessibility_step = ACCESSIBILITY_STEP = 1 # distance from agent to agent, defining neighbourhood

λ_ind = Λ_IND = 1. # individual learning
λ_wom = Λ_WOM = 1. # social learning
λ_ad = Λ_AD = 1.

#a = A = fill(1., NUM_FIRMS)
#σ_v0 = Σ_V0 = fill(0.5, 10)[1:NUM_FIRMS]
#σ_α = Σ_Α = fill(0.5, 10)[1:NUM_FIRMS]
#σ_ϵ = Σ_Ε = fill(0.5, 10)[1:NUM_FIRMS]

a = 1.
σ2_v0 = 0.5
σ2_ϵ = 0.5
σ2_α = 0.5

max_iter = MAX_ITER = 365

#ADVERTISING = fill(fill(0.0, MAX_ITER), NUM_FIRMS)

max_stock_period = MAX_STOCK_PERIOD = 10
wealth = WEALTH = 50.0
risk = RISK = 50.0
buyer_memory = BUYER_MEMORY = 15000

p = 60
c = 30
d = 10

discount, prices, margin = create_price(p,c,d,max_iter, num_firms)

cpp = 1.
its = 0.05
aits = 0.025

investment, advertising = create_advertising(cpp, its, aits, max_iter, num_firms, num_customers)

@time res_quality_uncertainty, res_quality_expectation, res_unit_bought, res_received_ad, res_neighbours_bought, res_stock = simulate(num_customers, num_firms, max_iter, prices, advertising, a, σ2_v0, σ2_ϵ, σ2_α, wealth, max_stock_period, risk, network_type, num_links, pref_attachment_links, accessibility_step, λ_ind, λ_wom, λ_ad, buyer_memory)

sum(res_unit_bought)

expectations = dropdims(mean(res_quality_expectation, dims=1), dims=1)
uncertainty = dropdims(mean(res_quality_uncertainty, dims=1), dims=1)

colors_red = [RGBA(1,0,0,(60 + i)/(max_iter+60)) for i in 1:max_iter]
colors_gre = [RGBA(0,1,0,(60 + i)/(max_iter+60)) for i in 1:max_iter]
colors_blu = [RGBA(0,0,1,(60 + i)/(max_iter+60)) for i in 1:max_iter]
colors_gb = [RGBA(0,1,1,(60 + i)/(max_iter+60)) for i in 1:max_iter]

p=Plots.plot(uncertainty[1,:], expectations[1,:], color = colors_red, xflip=true, xlabel = "Uncertainty", ylabel = "Expected quality", label = "Firm #1", legend = :bottomleft)
p=plot!(uncertainty[2,:], expectations[2,:], color = colors_gre, xflip=true, label = "Firm #2")
p=plot!(uncertainty[3,:], expectations[3,:], color = colors_blu, xflip=true, label = "Firm #3")
p=plot!(uncertainty[4,:], expectations[4,:], color = colors_gb, xflip=true, label = "Firm #4")

savefig(p, "C:/Users/mihau/Documents/system_dynamics_ad_discount.pdf")

std_uncertainty = dropdims(std(res_quality_uncertainty, dims=1), dims=1)

p9=plot(std_uncertainty[1,:], label = "Firm #1",xlabel="Iteration", ylabel="St.d. of uncertainty")
p9=plot!(std_uncertainty[2,:], label = "Firm #2")
p9=plot!(std_uncertainty[3,:], label = "Firm #3")
p9=plot!(std_uncertainty[4,:], label = "Firm #4")

savefig(p9, "C:/Users/mihau/Documents/system_dynamics_std_uncertainty.pdf")


Plots.plot(uncertainty[1,:])
Plots.plot!(uncertainty[2,:])
Plots.plot!(uncertainty[3,:])
Plots.plot!(uncertainty[4,:])



function circle_shape(h,k,r)
    θ = LinRange(0,2π,600)
    return h .+ r*sin.(θ), k .+ r*cos.(θ)
end

p2=scatter(uncertainty[:,10], expectations[:,10],color = ["red","green","blue","cyan"], label=nothing, xflip=true, legend=:bottomleft,xlabel = "Uncertainty", ylabel = "Expected quality")
p2=plot!(circle_shape(0.35,0.995,0.06), linecolor="black",linestyle=:dot, label = "Iteration: 10")

p2=scatter!(uncertainty[:,100], expectations[:,100],color = ["red","green","blue","cyan"], xflip=true, label=nothing)
p2=plot!(circle_shape(0.075,0.975,0.03), linecolor="black", linestyle=:dash, label = "Iteration: 100")


p2=scatter!(uncertainty[:,end], expectations[:,end],color = ["red","green","blue","cyan"], xflip=true, label=nothing)
p2=plot!(circle_shape(0.025,0.985,0.02), linecolor="black", label = "Iteration: 365")

savefig(p2, "C:/Users/mihau/Documents/system_dynamics_markovitz_dynamic.pdf")


buying = sum(res_unit_bought, dims=1)
buying = dropdims(buying, dims=1)
buying = buying[:,3:end]

sum(buying, dims=2)

p3 = groupedbar(transpose(buying), bar_position = :stack, lw=0, xlabel = "Time", ylabel = "Sales", color = ["red" "green" "blue" "cyan"], legend = nothing)

savefig(p3, "C:/Users/mihau/Documents/system_dynamics_sales.pdf")

p7=plot(res_quality_uncertainty[1,1,2:end], color=:lightgrey,legend=false)
[Plots.display(plot!(res_quality_uncertainty[i,1,2:end], color=:lightgrey,legend=false)) for i in 2:200]


plot!(uncertainty[1,:], color="red", lw=2)
xlabel!("Iteration")
ylabel!("Uncertainty of brand 1")

savefig(p7, "C:/Users/mihau/Documents/system_single_uncertainty.pdf")

"""