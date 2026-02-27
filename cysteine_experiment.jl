#!/usr/bin/env julia
#=
Cysteine enrichment steering experiment
========================================
Generates proteins with and without DTS* reward steering,
where the reward is 1 if the sequence contains more than 7
cysteine residues, and 0 otherwise. Produces a comparison plot.
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "ChainStorm.jl"))

using ChainStorm, Flowfusion, ForwardBackward

# ─── AA index → string conversion ────────────────────────────────────────────

const AA_CHARS = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']

function seq_string(aa_indices)
    return String([AA_CHARS[i] for i in aa_indices])
end

function extract_sequences(g)
    ints = tensor(Flowfusion.unhot(g[3]))
    return [seq_string(ints[:, b]) for b in axes(ints, 2)]
end

# ─── Cysteine detection ─────────────────────────────────────────────────────

cysteine_count(seq::String) = count(==('C'), seq)

has_enough_cys(seq::String) = cysteine_count(seq) > 7

function cysteine_reward(g)
    seqs = extract_sequences(g)
    return any(has_enough_cys, seqs) ? 1.0 : 0.0
end

# ─── Generation ──────────────────────────────────────────────────────────────

function generate_samples(; n_samples = 20, chain_lengths = [40, 40], steps = 30)
    model = load_model()

    unconditional_seqs = String[]
    conditional_seqs = String[]

    println("\n=== Generating $n_samples conditional (DTS*) samples (steps=$steps) ===")
    for i in 1:n_samples
        print("  Sample $i/$n_samples ")
        b = dummy_batch(chain_lengths)
        g = flow_treegen(b, model;
            reward = cysteine_reward,
            steps = steps,
            n_iterations = 20,
            max_children = 3,
            branching_points = [0.0, 0.05, 0.15, 0.4, 0.7],
            c_uct = 1.0,
            lambda = 2.0,
        )
        append!(conditional_seqs, extract_sequences(g))
    end

    println("\n=== Generating $n_samples unconditional samples (steps=$steps) ===")
    for i in 1:n_samples
        print("  Sample $i/$n_samples ")
        b = dummy_batch(chain_lengths)
        g = flow_quickgen(b, model; steps = steps)
        append!(unconditional_seqs, extract_sequences(g))
        println()
    end

    return unconditional_seqs, conditional_seqs
end

# ─── Analysis & Visualization ────────────────────────────────────────────────

function count_hits(seqs)
    n_with = count(has_enough_cys, seqs)
    cys_counts = [cysteine_count(s) for s in seqs]
    return (; n_with, frac = n_with / length(seqs), cys_counts)
end

function print_report(uncond_seqs, cond_seqs)
    u = count_hits(uncond_seqs)
    c = count_hits(cond_seqs)

    println("\n" * "="^60)
    println("       Cysteine Enrichment (>7 Cys) Report")
    println("="^60)
    println()
    println("  Unconditional:  $(u.n_with)/$(length(uncond_seqs)) sequences with >7 Cys ($(round(u.frac*100, digits=1))%)")
    println("  DTS* steered:   $(c.n_with)/$(length(cond_seqs)) sequences with >7 Cys ($(round(c.frac*100, digits=1))%)")
    println()

    mean_u = round(sum(u.cys_counts) / length(u.cys_counts), digits=2)
    mean_c = round(sum(c.cys_counts) / length(c.cys_counts), digits=2)
    println("  Mean Cys count:")
    println("    Unconditional: $mean_u")
    println("    DTS* steered:  $mean_c")
    println()

    # Show top examples
    for (label, seqs, stats) in [("Unconditional", uncond_seqs, u), ("DTS*", cond_seqs, c)]
        sorted = sort(collect(zip(stats.cys_counts, seqs)), by = x -> -x[1])
        println("  Top 3 $label by Cys count:")
        for (cnt, seq) in sorted[1:min(3, length(sorted))]
            println("    Cys=$cnt  $(seq[1:min(40,length(seq))])...")
        end
        println()
    end
    println("="^60)
end

function make_plot(uncond_seqs, cond_seqs; outfile = joinpath(@__DIR__, "cysteine_comparison.png"))
    u = count_hits(uncond_seqs)
    c = count_hits(cond_seqs)

    try
        @eval using CairoMakie

        fig = Figure(size = (700, 400), fontsize = 14)

        # Bar chart: fraction with >7 Cys
        ax1 = Axis(fig[1, 1],
            title = "Sequences with >7 Cys",
            ylabel = "Fraction",
            xticks = ([1, 2], ["Unconditional", "DTS*"]),
            ylabelsize = 13,
        )
        barplot!(ax1, [1, 2], [u.frac, c.frac],
            color = [:steelblue, :coral],
            strokewidth = 1, strokecolor = :black)
        ylims!(ax1, 0, max(c.frac * 1.3, u.frac * 1.3, 0.1))

        # Histogram: Cys count per sequence
        ax2 = Axis(fig[1, 2],
            title = "Cysteine count per sequence",
            xlabel = "# Cys residues",
            ylabel = "# sequences",
            ylabelsize = 13,
        )
        all_counts = vcat(u.cys_counts, c.cys_counts)
        max_count = max(maximum(all_counts; init=0), 1)
        bins = -0.5:1:(max_count + 0.5)
        hist!(ax2, u.cys_counts, bins = bins, color = (:steelblue, 0.6), label = "Unconditional", strokewidth = 1)
        hist!(ax2, c.cys_counts, bins = bins, color = (:coral, 0.6), label = "DTS*", strokewidth = 1)
        vlines!(ax2, [7.5], color = :red, linestyle = :dash, label = "Threshold (>7)")
        axislegend(ax2, position = :rt, framevisible = false)

        save(outfile, fig, px_per_unit = 2)
        println("\nPlot saved to: $outfile")
    catch e
        println("\n(CairoMakie not available: $e)")
        println("\n  Fraction with >7 Cys:")
        n = 40
        u_bar = round(Int, u.frac * n)
        c_bar = round(Int, c.frac * n)
        println("    Uncond  |$("█"^u_bar)$("░"^(n-u_bar))| $(round(u.frac*100, digits=1))%")
        println("    DTS*    |$("█"^c_bar)$("░"^(n-c_bar))| $(round(c.frac*100, digits=1))%")
    end
end

# ─── Main ────────────────────────────────────────────────────────────────────

println("Cysteine enrichment steering experiment (CPU, 20 samples, 30 steps)")
uncond, cond = generate_samples()
print_report(uncond, cond)
make_plot(uncond, cond)
