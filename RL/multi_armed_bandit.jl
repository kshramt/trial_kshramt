#!/usr/bin/env julia


module MultiArmedBandit

abstract type AbstractAgent end
abstract type AbstractEnv end


# ϵ-greedy
mutable struct EpsilonGreedyAgent{R<:AbstractFloat, I<:Integer} <: AbstractAgent
    ϵ::R
    n::I
    ns::Vector{I}
    μs::Vector{R}

    function EpsilonGreedyAgent{R, I}(ϵ, n, ns, μs) where {R<:AbstractFloat, I<:Integer}
        @assert 0 ≤ ϵ ≤ 1
        @assert n == 0
        @assert 0 < length(ns) == length(μs)
        @assert all(ns .== 0)
        @assert all(μs .== 0)
        new(ϵ, n, ns, μs)
    end
end

EpsilonGreedyAgent(ϵ::R, n_arms::I) where {R, I} = EpsilonGreedyAgent{R, I}(ϵ, zero(I), zeros(I, n_arms), zeros(R, n_arms))


function action_of(agent::EpsilonGreedyAgent, state)
    if agent.n < n_arms_of(agent)
        agent.n + 1
    else
        if rand() <= agent.ϵ
            rand(1:n_arms_of(agent))
        else
            indmax(agent.μs)
        end
    end
end


# UCB1
mutable struct UCB1Agent{R<:AbstractFloat, I<:Integer} <: AbstractAgent
    n::I
    ns::Vector{I}
    μs::Vector{R}

    function UCB1Agent{R, I}(n, ns, μs) where {R<:AbstractFloat, I<:Integer}
        @assert n == 0
        @assert 0 < length(ns) == length(μs)
        @assert all(ns .== 0)
        @assert all(μs .== 0)
        new(n, ns, μs)
    end
end

UCB1Agent(n_arms::I) where {I} = UCB1Agent{Float64, I}(zero(I), zeros(I, n_arms), zeros(Float64, n_arms))


function action_of(agent::UCB1Agent, state)
    if agent.n < n_arms_of(agent)
        agent.n + 1
    else
        indmax(agent.μs .+ sqrt.(2.*log.(agent.n)./agent.ns))
    end
end


struct Env <: AbstractEnv
    rngs::AbstractVector
end

reward_of(env::Env, action) = rand(env.rngs[action])

function act(env::Env, action)
    env, nothing, reward_of(env, action)
end


function learn(agent::AbstractAgent, state, action, reward)
    agent.n += 1
    nⱼ = (agent.ns[action] += 1)
    agent.μs[action] = (nⱼ - 1)*agent.μs[action]/nⱼ + reward/nⱼ
    agent
end

n_arms_of(agent::AbstractAgent) = length(agent.ns)

end # module


using Distributions

const mab = MultiArmedBandit


function main(args)
    μs = linspace(0.0, 1.0, 10)
    rngs = map(Distributions.Bernoulli, μs)
    env = mab.Env(rngs)

    agent = mab.UCB1Agent(length(rngs))
    # agent = mab.EpsilonGreedyAgent(0.05, length(rngs))

    stateₜ = nothing
    action = mab.action_of(agent, stateₜ)
    for _ in 1:1000 # while true
        env, stateₜ₊₁, reward = mab.act(env, action)
        # @show reward agent.μs agent.ns
        println(reward)
        agent = mab.learn(agent, stateₜ, action, reward)
        action = mab.action_of(agent, stateₜ₊₁)
        stateₜ = stateₜ₊₁
    end
end


function _usage_and_exit(s=1)
    io = s == 0 ? STDOUT : STDERR
    println(io, "$PROGRAM_FILE > out.txt")
    exit(s)
end


if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
