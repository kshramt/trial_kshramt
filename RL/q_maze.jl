#!/usr/bin/env julia


using FileIO


const maze = [
    -1 -1 -1 -1 -1 -1 -1
    -1  0  0 -1  0  1 -1
    -1  0  0 -1  0  0 -1
    -1  0  0 -1  0  0 -1
    -1  0  0  0  0  0 -1
    -1  0  0 -1  0  0 -1
    -1 -1 -1 -1 -1 -1 -1
]


function main(args)
    go(maze, args[1], parse(Int, args[2]))
end


function _usage_and_exit(s=1)
    io = s == 0 ? STDOUT : STDERR
    println(io, "$PROGRAM_FILE <out.jld2> <seed>")
    exit(s)
end


function go(maze, out, seed)
    srand(seed)
    γ = 0.95
    α = 0.05
    Q = zeros(Float64, (size(maze)..., 4))

    i_ep = 0
    while true
        i_ep += 1
        if i_ep > 1000
            break
        end
        episode(maze, Q, γ, α)
    end

    mkpath(dirname(out))
    FileIO.save(out, "Q", Q, "maze", maze)
end


function episode(maze, Q, γ, α)
    sᵢ = P(maze)
    i_iter = 0

    # s0 -> a1 -> (r1, s1) -> a2
    while true
        i_iter += 1
        (i_iter > 100000) && break
        aᵢ₊₁ = pai(sᵢ, maze)
        rᵢ₊₁, sᵢ₊₁ = act(maze, sᵢ, aᵢ₊₁)
        terminate = maze[sᵢ₊₁...] == 1
        TD = (rᵢ₊₁ + (terminate ? zero(γ) : γ*maximum(Q[sᵢ₊₁..., :]))) - Q[sᵢ..., aᵢ₊₁]
        println(TD)
        Q[sᵢ..., aᵢ₊₁] += α*TD
        terminate && break
        sᵢ = sᵢ₊₁
    end
    println()
end


function act(env, s, a)
    n, m = size(env)
    i, j = s
    s = if (a == 1) && (j < m)
        (i, j + 1)
    elseif (a == 2) && (i > 1)
        (i - 1, j)
    elseif (a == 3) && (j > 1)
        (i, j - 1)
    elseif (a == 4) && (i < n)
        (i + 1, j)
    else # stopped by a wall
        (i, j)
    end
    r = env[s...]
    return r, s
end


"""
 2
3 1
 4
"""
function pai(s, env)
    N, M = size(maze)
    i, j = s
    while true
        a = rand(1:4)
        (i == 1 && a == 2) || (i == N && a == 4) || (j == 1 && a == 3) || (j == M && a == 1) || return a
    end
end


function P(maze)
    @assert 0 in maze
    N, M = size(maze)
    while true
        i, j = rand(1:N), rand(1:M)
        if maze[i, j] == 0
            return i, j
        end
    end
end


if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
