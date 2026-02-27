#!/usr/bin/env julia
#=
BranchChain cysteine enrichment steering experiment (fraction reward)
=====================================================================
Generates proteins with and without DTS* reward steering.
Reward is cysteine fraction among redesigned residues only:
    reward = (# C) / (# non-X residues)
This prevents reward hacking by merely extending sequence length.
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "BranchChain.jl"))

include(joinpath(@__DIR__, "branchchain_benchmark_utils.jl"))
using .BranchChainBench

cysteine_count(seq::String) = count(==('C'), seq)
residue_count(seq::String) = count(!=('X'), seq)
cysteine_fraction(seq::String) = BranchChainBench.safe_fraction(cysteine_count(seq), residue_count(seq))
has_enough_cys_fraction(seq::String; threshold = 7 / 40) = cysteine_fraction(seq) > threshold

cysteine_reward(seq::String) = cysteine_fraction(seq)

function generate_samples(; n_samples = 20, steps = 30)
    return BranchChainBench.generate_samples(
        reward = cysteine_reward,
        n_samples = n_samples,
        steps = steps,
        n_iterations = 20,
        max_children = 3,
        branching_points = [0.0, 0.05, 0.15, 0.4, 0.7],
        c_uct = 1.0,
        lambda = 10.0,
    )
end

function count_hits(seqs)
    n_with = count(s -> has_enough_cys_fraction(s), seqs)
    fractions = [cysteine_fraction(s) for s in seqs]
    counts = [cysteine_count(s) for s in seqs]
    lengths = [residue_count(s) for s in seqs]
    return (; n_with, frac = n_with / length(seqs), fractions, counts, lengths)
end

function print_report(uncond_seqs, cond_seqs)
    u = count_hits(uncond_seqs)
    c = count_hits(cond_seqs)

    mean_u_frac = round(100 * sum(u.fractions) / length(u.fractions), digits = 2)
    mean_c_frac = round(100 * sum(c.fractions) / length(c.fractions), digits = 2)
    mean_u_cnt = round(sum(u.counts) / length(u.counts), digits = 2)
    mean_c_cnt = round(sum(c.counts) / length(c.counts), digits = 2)
    mean_u_len = round(sum(u.lengths) / length(u.lengths), digits = 2)
    mean_c_len = round(sum(c.lengths) / length(c.lengths), digits = 2)

    println("\n" * "="^64)
    println("   BranchChain Cysteine Enrichment (fraction reward, redesigned only)")
    println("="^64)
    println()
    println("  Mean cysteine fraction:")
    println("    Unconditional: $(mean_u_frac)%")
    println("    DTS* steered:  $(mean_c_frac)%")
    println()
    println("  Mean cysteine count:")
    println("    Unconditional: $mean_u_cnt")
    println("    DTS* steered:  $mean_c_cnt")
    println()
    println("  Mean redesigned length (non-X):")
    println("    Unconditional: $mean_u_len")
    println("    DTS* steered:  $mean_c_len")
    println()
    println("  Fraction above threshold (> 7/40 = 17.5% Cys):")
    println("    Unconditional: $(u.n_with)/$(length(uncond_seqs)) ($(round(100 * u.frac, digits = 1))%)")
    println("    DTS* steered:  $(c.n_with)/$(length(cond_seqs)) ($(round(100 * c.frac, digits = 1))%)")
    println()

    for (label, seqs, stats) in [("Unconditional", uncond_seqs, u), ("DTS*", cond_seqs, c)]
        ranked = sort(collect(zip(stats.fractions, stats.counts, seqs)), by = x -> -x[1])
        println("  Top 3 $label by cysteine fraction:")
        for (frac, cnt, seq) in ranked[1:min(3, length(ranked))]
            pct = round(100 * frac, digits = 2)
            preview = seq[1:min(40, length(seq))]
            println("    Cys=$cnt  frac=$(pct)%  $(preview)...")
        end
        println()
    end
    println("="^64)
end

function make_plot(uncond_seqs, cond_seqs; outfile = joinpath(@__DIR__, "branchchain_cysteine_comparison.png"))
    u = count_hits(uncond_seqs)
    c = count_hits(cond_seqs)

    mean_u = sum(u.fractions) / length(u.fractions)
    mean_c = sum(c.fractions) / length(c.fractions)

    try
        @eval using CairoMakie

        fig = Figure(size = (760, 420), fontsize = 14)

        ax1 = Axis(fig[1, 1],
            title = "Mean Cysteine Fraction (redesigned residues)",
            ylabel = "Fraction",
            xticks = ([1, 2], ["Unconditional", "DTS*"]),
            ylabelsize = 13,
        )
        barplot!(ax1, [1, 2], [mean_u, mean_c],
            color = [:steelblue, :coral],
            strokewidth = 1, strokecolor = :black)
        ylims!(ax1, 0, max(mean_u, mean_c, 0.1) * 1.3)

        ax2 = Axis(fig[1, 2],
            title = "Cysteine Fraction per Sequence",
            xlabel = "Fraction cysteine (0-1)",
            ylabel = "# sequences",
            ylabelsize = 13,
        )
        bins = 0:0.025:1.0
        hist!(ax2, u.fractions, bins = bins, color = (:steelblue, 0.6), label = "Unconditional", strokewidth = 1)
        hist!(ax2, c.fractions, bins = bins, color = (:coral, 0.6), label = "DTS*", strokewidth = 1)
        axislegend(ax2, position = :rt, framevisible = false)

        save(outfile, fig, px_per_unit = 2)
        println("\nPlot saved to: $outfile")
    catch e
        println("\n(CairoMakie not available: $e)")
        println("\n  Mean cysteine fraction:")
        n = 40
        scale = max(mean_u, mean_c, 1e-6)
        u_bar = round(Int, mean_u / scale * n)
        c_bar = round(Int, mean_c / scale * n)
        println("    Uncond  |$(repeat('#', u_bar))$(repeat('-', n - u_bar))| $(round(100 * mean_u, digits = 2))%")
        println("    DTS*    |$(repeat('#', c_bar))$(repeat('-', n - c_bar))| $(round(100 * mean_c, digits = 2))%")
    end
end

function main()
    println("BranchChain cysteine enrichment experiment (CPU, 20 samples, 30 steps)")
    uncond, cond = generate_samples()
    print_report(uncond, cond)
    make_plot(uncond, cond)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
