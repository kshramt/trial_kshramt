#!/usr/bin/env julia

using FileIO
# using PyPlot


function main(args)
    data = FileIO.load(args[1])
    Q, maze = data["Q"], data["maze"]
    for i in 1:size(Q, 1)
        for j in 1:size(Q, 2)
            print("    ")
            @printf "%6.2f " Q[i, j, 2]
            print("    ")
        end
        print("\n")
        for j in 1:size(Q, 2)
            @printf "%6.2f " Q[i, j, 3]
            print(" ")
            @printf "%6.2f " Q[i, j, 1]
        end
        print("\n")
        for j in 1:size(Q, 2)
            print("    ")
            @printf "%6.2f " Q[i, j, 4]
            print("    ")
        end
        print("\n")

        print("\n")
    end
end


function _usage_and_exit(s=1)
    io = s == 0 ? STDOUT : STDERR
    println(io, "$PROGRAM_FILE <in.jld2>")
    exit(s)
end


if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
