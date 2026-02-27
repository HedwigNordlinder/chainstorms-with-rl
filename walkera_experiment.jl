#!/usr/bin/env julia
#=
Walker-A motif steering experiment
===================================
Generates proteins with and without DTS* reward steering,
where the reward is 1 if the sequence contains a Walker-A-like motif (G.{4}GK[TS]),
and 0 otherwise. Produces a comparison plot.
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

# ─── Walker-A motif detection ────────────────────────────────────────────────

const WALKER_A = r"G.{4}GK[TS]"

has_walker_a(seq::String) = occursin(WALKER_A, seq)

function walker_a_reward(g)
    seqs = extract_sequences(g)
    return any(has_walker_a, seqs) ? 1.0 : 0.0
end

# ─── Generation ──────────────────────────────────────────────────────────────

function generate_samples(; n_samples = 5, chain_lengths = [40, 40], steps = 50)
    model = load_model()

    unconditional_seqs = String[]
    conditional_seqs = String[]

    println("\n=== Generating $n_samples conditional (DTS*) samples (steps=$steps) ===")
    for i in 1:n_samples
        print("  Sample $i/$n_samples ")
        b = dummy_batch(chain_lengths)
        g = flow_treegen(b, model;
            reward = walker_a_reward,
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

function count_motifs(seqs)
    n_with = count(has_walker_a, seqs)
    matches_per_seq = [length(collect(eachmatch(WALKER_A, s))) for s in seqs]
    return (; n_with, frac = n_with / length(seqs), matches_per_seq)
end

function find_motif_positions(seq)
    [(m.offset, m.offset + length(m.match) - 1, m.match) for m in eachmatch(WALKER_A, seq)]
end

function print_report(uncond_seqs, cond_seqs)
    u = count_motifs(uncond_seqs)
    c = count_motifs(cond_seqs)

    println("\n" * "="^60)
    println("         Walker-A Motif (G.{4}GK[TS]) Report")
    println("="^60)
    println()
    println("  Unconditional:  $(u.n_with)/$(length(uncond_seqs)) sequences ($(round(u.frac*100, digits=1))%)")
    println("  DTS* steered:   $(c.n_with)/$(length(cond_seqs)) sequences ($(round(c.frac*100, digits=1))%)")
    println()

    total_u = sum(u.matches_per_seq)
    total_c = sum(c.matches_per_seq)
    println("  Total motif occurrences:")
    println("    Unconditional: $total_u  ($(round(total_u/length(uncond_seqs), digits=3)) per seq)")
    println("    DTS* steered:  $total_c  ($(round(total_c/length(cond_seqs), digits=3)) per seq)")
    println()

    # Show example hits
    for (label, seqs) in [("Unconditional", uncond_seqs), ("DTS*", cond_seqs)]
        hits = [(s, find_motif_positions(s)) for s in seqs if has_walker_a(s)]
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

function make_plot(uncond_seqs, cond_seqs; outfile = joinpath(@__DIR__, "walkera_comparison.png"))
    u_matches = [length(collect(eachmatch(WALKER_A, s))) for s in uncond_seqs]
    c_matches = [length(collect(eachmatch(WALKER_A, s))) for s in cond_seqs]

    u_frac = count(has_walker_a, uncond_seqs) / length(uncond_seqs)
    c_frac = count(has_walker_a, cond_seqs) / length(cond_seqs)

    try
        @eval using CairoMakie

        fig = Figure(size = (700, 400), fontsize = 14)

        # Bar chart: fraction with motif
        ax1 = Axis(fig[1, 1],
            title = "Sequences with Walker-A motif",
            ylabel = "Fraction",
            xticks = ([1, 2], ["Unconditional", "DTS*"]),
            ylabelsize = 13,
        )
        barplot!(ax1, [1, 2], [u_frac, c_frac],
            color = [:steelblue, :coral],
            strokewidth = 1, strokecolor = :black)
        ylims!(ax1, 0, max(c_frac * 1.3, u_frac * 1.3, 0.1))

        # Histogram: motif count per sequence
        ax2 = Axis(fig[1, 2],
            title = "Motif count per sequence",
            xlabel = "# Walker-A motifs",
            ylabel = "# sequences",
            ylabelsize = 13,
        )
        max_count = max(maximum(u_matches; init=0), maximum(c_matches; init=0), 1)
        bins = -0.5:1:(max_count + 0.5)
        hist!(ax2, u_matches, bins = bins, color = (:steelblue, 0.6), label = "Unconditional", strokewidth = 1)
        hist!(ax2, c_matches, bins = bins, color = (:coral, 0.6), label = "DTS*", strokewidth = 1)
        axislegend(ax2, position = :rt, framevisible = false)

        save(outfile, fig, px_per_unit = 2)
        println("\nPlot saved to: $outfile")
    catch e
        println("\n(CairoMakie not available: $e)")
        println("\n  Fraction with Walker-A motif:")
        n = 40
        u_bar = round(Int, u_frac * n)
        c_bar = round(Int, c_frac * n)
        println("    Uncond  |$("█"^u_bar)$("░"^(n-u_bar))| $(round(u_frac*100, digits=1))%")
        println("    DTS*    |$("█"^c_bar)$("░"^(n-c_bar))| $(round(c_frac*100, digits=1))%")
    end
end

# ─── Main ────────────────────────────────────────────────────────────────────

println("Walker-A motif steering experiment (CPU, 5 samples, 50 steps)")
uncond, cond = generate_samples()
print_report(uncond, cond)
make_plot(uncond, cond)
