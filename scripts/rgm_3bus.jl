using NLsolve
using ForwardDiff
using LinearAlgebra
using Printf
using Plots
include("../src/algorithms/linesearches.jl")

# ELEMENTAL FUNCTIONS

function gfun!(F, x, u, p)
  
  # retrieve variables
  VM3 = x[1]
  VA3 = x[2]
  VA2 = x[3]

  VM1 = u[1]
  P2 = u[2]
  VM2 = u[3]

  VA1 = p[1]
  P3 = p[2]
  Q3 = p[3]

  # intermediate quantities
  VA23 = VA2 - VA3
  VA31 = VA3 - VA1
  VA32 = VA3 - VA2
  VA13 = VA1 - VA3
  VA21 = VA2 - VA1
  VA12 = VA1 - VA2

  PD2 = 3.0
  PD3 = 2.0

  F[1] = (VM2*VM1*(-1*cos(VA21) + 10*sin(VA21))
          + 3.0*VM2*VM2
          + VM2*VM3*(-2*cos(VA23) + 20*sin(VA23))
          + PD2
          - P2)
  F[2] = (VM3*VM1*(-1*cos(VA31) + 8.0*sin(VA31))
          + VM3*VM2*(-2*cos(VA32) + 20.0*sin(VA32))
          + 3.0*VM3*VM3
          + PD3)
  F[3] = (VM3*VM1*(-1*sin(VA31) - 8.0*cos(VA31))
          + VM3*VM2*(-2*sin(VA32) - 20.0*cos(VA32))
          + 28.0*VM3*VM3)
end

function gfun(x, u, p, T=typeof(x))
  # Get Float64 of Vectors{Float64}, that is the first parameter
  F = zeros(T.parameters[1], 3)

  # retrieve variables
  VM3 = x[1]
  VA3 = x[2]
  VA2 = x[3]

  VM1 = u[1]
  P2 = u[2]
  VM2 = u[3]

  VA1 = p[1]
  P3 = p[2]
  Q3 = p[3]

  # intermediate quantities
  VA23 = VA2 - VA3
  VA31 = VA3 - VA1
  VA32 = VA3 - VA2
  VA13 = VA1 - VA3
  VA21 = VA2 - VA1
  VA12 = VA1 - VA2

  PD2 = 3.0
  PD3 = 2.0

  F[1] = (VM2*VM1*(-1*cos(VA21) + 10*sin(VA21))
          + 3.0*VM2*VM2
          + VM2*VM3*(-2*cos(VA23) + 20*sin(VA23))
          + PD2
          - P2)
  F[2] = (VM3*VM1*(-1*cos(VA31) + 8.0*sin(VA31))
          + VM3*VM2*(-2*cos(VA32) + 20.0*sin(VA32))
          + 3.0*VM3*VM3
          + PD3)
  F[3] = (VM3*VM1*(-1*sin(VA31) - 8.0*cos(VA31))
          + VM3*VM2*(-2*sin(VA32) - 20.0*cos(VA32))
          + 28.0*VM3*VM3)
  
  return F
end


function cfun(x, u, p)

  VM3 = x[1]
  VA3 = x[2]
  VA2 = x[3]

  VM1 = u[1]
  P2 = u[2]
  VM2 = u[3]

  VA1 = p[1]

  VA13 = VA1 - VA3
  VA12 = VA1 - VA2

  bmva = 100.0
  P1 = (2.0*VM1*VM1
          + VM1*VM2*(-1*cos(VA12) + 10.0*sin(VA12))
          + VM1*VM3*(-1*cos(VA13) + 8*sin(VA13)))
  cost = 0.6 + bmva*P1 + bmva*2.0*P2 + bmva*bmva*P1*P1 + bmva*bmva*0.5*P2*P2
  
  return cost
end

# OPF COMPUTATION

function solve_pf(x, u, p, verbose=true)
 
  fun_pf!(F, x) = gfun!(F, x, u, p)
  x0 = copy(x)
  res = nlsolve(fun_pf!, x0)
  if verbose
    show(res)
    println("")
  end
  
  return res.zero

end

# initial parameters
x = zeros(3)
u = zeros(3)
p = zeros(3)

# this is an initial guess
x[1] = 1.0 #VM3
x[2] = 0.0 #VA3
x[3] = 0.0 #VA2

# this is given by the problem data, but might be "controlled" via OPF
u[1] = 1.0 #VM1
u[2] = 2.0 #P2
u[3] = 1.0 #VM2

# these parameters are fixed through the computation
p[1] = 0.0 #VA1, slack angle
p[2] = 0.0 #P3
p[3] = 0.0 #Q3



# print initial guesses
println(x)
println(u)

# copy to iteration vectors
xk = copy(x)
uk = copy(u)

iterations = 0


xk = solve_pf(xk, uk, p, false)


for i = 1:80
  global xk
  global uk
  println("Iteration ", i)

  # solve power flow
  println("Solving power flow")
  xk = solve_pf(xk, uk, p, false)
  println("Cost: ", cfun(xk, uk, p))

  # print
  println(xk)
  println(uk)

  # jacobiana
  gx_x(x) = gfun(x, uk, p, typeof(x))
  gx = x -> ForwardDiff.jacobian(gx_x, x)

  # gradient
  cfun_x(x) = cfun(x, uk, p)
  fx = x ->ForwardDiff.gradient(cfun_x, x)

  # lamba calculation
  println("Computing Lagrange multipliers")
  lambda = -inv(gx(xk)')*fx(xk)

  # compute g_u, g_u
  cfun_u(u) = cfun(xk, u, p)
  fu = u ->ForwardDiff.gradient(cfun_u, u)
  gx_u(u) = gfun(xk, u, p, typeof(u))
  gu = u -> ForwardDiff.jacobian(gx_u, u)

  # compute gradient of cost function
  Lu = u -> cfun_u(u) + gx_u(u)'*lambda
  grad_Lu = u -> (fu(u) + gu(u)'*lambda)
  grad_L = grad_Lu(uk)
  println("Norm of gradient ", norm(grad_L))
  println(grad_L)

  # step
  println("Computing new control vector")
  c_par = 0.1
  # Optional linesearch
  c_par = ls(uk, grad_L, Lu, grad_Lu)
  println("cpar: ", c_par)
  uk = uk - c_par*grad_L

  # Projection
  uk[1] = max(0.9, uk[1])
  uk[3] = max(0.9, uk[3])
  uk[1] = min(1.1, uk[1])
  uk[3] = min(1.1, uk[3])


  #@printf("VM3 %3.2f. VA3 %2.2f. VA2 %2.2f.\n", xk[1], xk[2], xk[3])
  #@printf("VM1 %3.2f. P2 %2.2f. VM2 %2.2f.\n", uk[1], uk[2], uk[3])
end
