#!/usr/bin/env julia
#=
Cysteine count distribution in unconditional samples
=====================================================
Generates 30 unconditional protein samples and shows the
distribution of cysteine (C) residues per sequence.
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

# ─── Generation ──────────────────────────────────────────────────────────────

function generate_unconditional(; n_samples = 30, chain_lengths = [40, 40], steps = 30)
    model = load_model()
    all_seqs = String[]

    println("=== Generating $n_samples unconditional samples (steps=$steps) ===")
    for i in 1:n_samples
        print("  Sample $i/$n_samples\n")
        b = dummy_batch(chain_lengths)
        g = flow_quickgen(b, model; steps = steps)
        append!(all_seqs, extract_sequences(g))
    end

    return all_seqs
end

# ─── Analysis ────────────────────────────────────────────────────────────────

function cysteine_count(seq::String)
    return count(==('C'), seq)
end

function print_cysteine_report(seqs)
    counts = [cysteine_count(s) for s in seqs]

    println("\n" * "="^50)
    println("    Cysteine Count Distribution ($(length(seqs)) sequences)")
    println("="^50)
    println()
    println("  Mean:   $(round(sum(counts)/length(counts), digits=2))")
    println("  Min:    $(minimum(counts))")
    println("  Max:    $(maximum(counts))")
    println("  Median: $(round(sort(counts)[length(counts)÷2 + 1], digits=1))")
    println()

    # ASCII histogram
    max_count = maximum(counts)
    bins = 0:max_count
    hist = zeros(Int, max_count + 1)
    for c in counts
        hist[c + 1] += 1
    end

    max_bar = maximum(hist)
    bar_width = 30
    println("  Count | # sequences")
    println("  ------+--" * "-"^bar_width)
    for (i, h) in enumerate(hist)
        c = i - 1
        bar_len = max_bar > 0 ? round(Int, h / max_bar * bar_width) : 0
        println("    $(lpad(c, 3))  | $("█"^bar_len) $h")
    end
    println()
    println("="^50)
end

# ─── Main ────────────────────────────────────────────────────────────────────

seqs = generate_unconditional()
print_cysteine_report(seqs)
