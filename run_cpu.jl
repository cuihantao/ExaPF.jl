using Revise
target = "cpu"
datafile = "SyntheticUSA/SyntheticUSA.RAW"
# # datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_30R-025/scenario_1/case.raw"
# datafile = "GO-Data/datasets/Trial_3_Real-Time/Network_13R-015/scenario_11/case.raw"
include("examples/pf.jl")
sol, conv, res = pf(datafile)
# sol, conv, res = pf(datafile, 84, "bicgstab")
# sol, conv, res = pf(datafile, 500)