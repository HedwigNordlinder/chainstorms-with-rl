#!/usr/bin/env julia
#=
BranchChain FxF motif steering experiment
=========================================
Generates proteins with and without DTS* reward steering.
Reward is 1 if redesigned residues contain F.F (Phe-X-Phe), else 0.
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "BranchChain.jl"))

include(joinpath(@__DIR__, "branchchain_benchmark_utils.jl"))
using .BranchChainBench

const FXF_MOTIF = r"F.F"

has_fxf(seq::String) = occursin(FXF_MOTIF, seq)
fxf_reward(seq::String) = has_fxf(seq) ? 1.0 : 0.0

function generate_samples(; n_samples = 10, steps = 30)
    return BranchChainBench.generate_samples(
        reward = fxf_reward,
        n_samples = n_samples,
        steps = steps,
        n_iterations = 20,
        max_children = 3,
        branching_points = [0.0, 0.05, 0.15, 0.4, 0.7],
        c_uct = 1.0,
        lambda = 2.0,
    )
end

function count_motifs(seqs)
    n_with = count(has_fxf, seqs)
    matches_per_seq = [length(collect(eachmatch(FXF_MOTIF, s))) for s in seqs]
    return (; n_with, frac = n_with / length(seqs), matches_per_seq)
end

function find_motif_positions(seq)
    [(m.offset, m.offset + length(m.match) - 1, m.match) for m in eachmatch(FXF_MOTIF, seq)]
end

function print_report(uncond_seqs, cond_seqs)
    u = count_motifs(uncond_seqs)
    c = count_motifs(cond_seqs)

    println("\n" * "="^60)
    println("     BranchChain FxF Motif (F.F / Phe-X-Phe) Report")
    println("="^60)
    println()
    println("  Unconditional:  $(u.n_with)/$(length(uncond_seqs)) sequences ($(round(u.frac * 100, digits = 1))%)")
    println("  DTS* steered:   $(c.n_with)/$(length(cond_seqs)) sequences ($(round(c.frac * 100, digits = 1))%)")
    println()

    total_u = sum(u.matches_per_seq)
    total_c = sum(c.matches_per_seq)
    println("  Total motif occurrences:")
    println("    Unconditional: $total_u  ($(round(total_u / length(uncond_seqs), digits = 3)) per seq)")
    println("    DTS* steered:  $total_c  ($(round(total_c / length(cond_seqs), digits = 3)) per seq)")
    println()

    for (label, seqs) in [("Unconditional", uncond_seqs), ("DTS*", cond_seqs)]
        hits = [(s, find_motif_positions(s)) for s in seqs if has_fxf(s)]
        if !isempty(hits)
            println("  Example $label hits (up to 3):")
            for (seq, positions) in hits[1:min(3, length(hits))]
                for (lo, hi, motif) in positions
                    ctx_lo = max(1, lo - 3)
                    ctx_hi = min(length(seq), hi + 3)
                    context = seq[ctx_lo:ctx_hi]
                    println("    ...$(context)...  (motif '$motif' at pos $lo-$hi)")
                end
            end
            println()
        end
    end
    println("="^60)
end

function make_plot(uncond_seqs, cond_seqs; outfile = joinpath(@__DIR__, "branchchain_fxf_comparison.png"))
    u_matches = [length(collect(eachmatch(FXF_MOTIF, s))) for s in uncond_seqs]
    c_matches = [length(collect(eachmatch(FXF_MOTIF, s))) for s in cond_seqs]

    u_frac = count(has_fxf, uncond_seqs) / length(uncond_seqs)
    c_frac = count(has_fxf, cond_seqs) / length(cond_seqs)

    try
        @eval using CairoMakie

        fig = Figure(size = (700, 400), fontsize = 14)

        ax1 = Axis(fig[1, 1],
            title = "Sequences with FxF Motif",
            ylabel = "Fraction",
            xticks = ([1, 2], ["Unconditional", "DTS*"]),
            ylabelsize = 13,
        )
        barplot!(ax1, [1, 2], [u_frac, c_frac],
            color = [:steelblue, :coral],
            strokewidth = 1, strokecolor = :black)
        ylims!(ax1, 0, max(c_frac * 1.3, u_frac * 1.3, 0.1))

        ax2 = Axis(fig[1, 2],
            title = "Motif Count per Sequence",
            xlabel = "# FxF motifs",
            ylabel = "# sequences",
            ylabelsize = 13,
        )
        max_count = max(maximum(u_matches; init = 0), maximum(c_matches; init = 0), 1)
        bins = -0.5:1:(max_count + 0.5)
        hist!(ax2, u_matches, bins = bins, color = (:steelblue, 0.6), label = "Unconditional", strokewidth = 1)
        hist!(ax2, c_matches, bins = bins, color = (:coral, 0.6), label = "DTS*", strokewidth = 1)
        axislegend(ax2, position = :rt, framevisible = false)

        save(outfile, fig, px_per_unit = 2)
        println("\nPlot saved to: $outfile")
    catch e
        println("\n(CairoMakie not available: $e)")
        println("\n  Fraction with FxF motif:")
        n = 40
        u_bar = round(Int, u_frac * n)
        c_bar = round(Int, c_frac * n)
        println("    Uncond  |$(repeat('#', u_bar))$(repeat('-', n - u_bar))| $(round(u_frac * 100, digits = 1))%")
        println("    DTS*    |$(repeat('#', c_bar))$(repeat('-', n - c_bar))| $(round(c_frac * 100, digits = 1))%")
    end
end

function main()
    println("BranchChain FxF motif steering experiment (CPU, 10 samples, 30 steps)")
    uncond, cond = generate_samples()
    print_report(uncond, cond)
    make_plot(uncond, cond)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
