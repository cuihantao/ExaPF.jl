# using Revise
target = "cuda"
datafile = "test/case14.raw"
blocks = parse(Int, ARGS[1])
println("Running PF with $blocks blocks")
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_30R-025/scenario_1/case.raw"
# datafile = "SyntheticUSA/SyntheticUSA.RAW"
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
include("examples/pf.jl")
sol, conv, res = pf(datafile, blocks, "bicgstab")
# sol, conv, res = pf(datafile, 1000, "bicgstab")