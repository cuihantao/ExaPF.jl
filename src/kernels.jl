module kernels
  using CUDAnative, CuArrays
  export sync, dispatch, generate, togenerate, angle

  togenerate = "cpu"

  macro angle(val)
    ex = nothing
    if togenerate == "cuda"
      ex = quote 
        CUDAnative.angle.($val)
      end
    end
    if togenerate == "cpu"
      ex = quote 
        angle.($val)
      end
    end
    return esc(ex)
  end

  macro cos(val)
    ex = nothing
    if togenerate == "cuda"
      ex = quote 
        CUDAnative.cos($val)
      end
    end
    if togenerate == "cpu"
      ex = quote 
        cos($val)
      end
    end
    return esc(ex)
  end

  macro sin(val)
    ex = nothing
    if togenerate == "cuda"
      ex = quote 
        CUDAnative.sin($val)
      end
    end
    if togenerate == "cpu"
      ex = quote 
        sin($val)
      end
    end
    return esc(ex)
  end

  macro sync(expr)
    ex = nothing
    if togenerate == "cuda"
      ex = quote 
        CuArrays.@sync begin
        $expr
        end
      end
    end
    if togenerate == "cpu"
      ex = quote 
        $expr
      end
    end
    return esc(ex)
  end

  macro dispatch(threads, blocks, expr)
    ex = nothing
    if togenerate == "cuda"
      ex = quote 
        @cuda $threads $blocks $expr
      end
    end
    if togenerate == "cpu"
      ex = quote 
        $expr
      end
    end
    return esc(ex)
  end

  macro getstrideindex()
    ex = nothing
    if togenerate == "cuda"
      ex = quote 
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = blockDim().x * gridDim().x
      end
    end
    if togenerate == "cpu"
      ex = quote
        index  = 1
        stride = 1
      end
    end
    return esc(ex)
  end
end