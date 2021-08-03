v = 0.9
s = 0.02

f(v,s) = v / (v^2 + s^2)
f(v,s)

using Plots; pyplot()
v=range(0.1,stop=2,length=100)
s=range(0.1,stop=2,length=100)
f(v,s) = v / (0.25(v^2 + s^2))
plot(v,s,f,st=:surface,camera=(-210,30), xlabel = "V", ylabel = "S", zlabel = "R")
f(2,2)5
