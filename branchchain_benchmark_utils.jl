#!/usr/bin/env julia

module BranchChainBench

using BranchChain
using ProteinChains: readpdb
using DLProteinFormats

const DEFAULT_TEMPLATE_PATH = joinpath(@__DIR__, "15-pgdh.pdb")
const DEFAULT_CHECKPOINT = "branchchain_tune1.jld"
const DEFAULT_BRANCHING_POINTS = [0.0, 0.05, 0.15, 0.4, 0.7]

safe_fraction(num::Real, den::Real) = den == 0 ? 0.0 : num / den

function load_template(path::AbstractString = DEFAULT_TEMPLATE_PATH)
    isfile(path) || error("Template PDB not found: $path")
    return readpdb(path)
end

function full_flattened_sequence(template)
    template.cluster = 1
    rec = DLProteinFormats.flatten(template)
    return join(DLProteinFormats.AAs[rec.AAs])
end

function make_design_target(template)
    full_seq = full_flattened_sequence(template)
    return X1_from_pdb(template, [full_seq])
end

function redesigned_sequence(state)
    aas = Int.(BranchChain.tensor(state.state[3]))
    mask = Bool.(state.flowmask)
    n = min(length(aas), length(mask))
    n == 0 && return ""
    keep = findall(mask[1:n])
    isempty(keep) && return ""
    return DLProteinFormats.ints_to_aa(aas[keep])
end

function generate_samples(;
    reward,
    n_samples::Int,
    steps,
    n_iterations::Int = 20,
    max_children::Int = 3,
    branching_points = DEFAULT_BRANCHING_POINTS,
    c_uct::Float64 = 1.0,
    lambda::Float64 = 1.0,
    checkpoint::String = DEFAULT_CHECKPOINT,
    template_path::String = DEFAULT_TEMPLATE_PATH,
)
    model = load_model(checkpoint)
    template = load_template(template_path)

    unconditional_seqs = String[]
    conditional_seqs = String[]

    println("\n=== Generating $n_samples conditional (DTS*) samples (steps=$steps) ===")
    for i in 1:n_samples
        print("  Sample $i/$n_samples ")
        x1 = make_design_target(template)
        g = design_treegen(model, x1;
            reward = s -> Float64(reward(redesigned_sequence(s))),
            steps = steps,
            n_iterations = n_iterations,
            max_children = max_children,
            branching_points = branching_points,
            c_uct = c_uct,
            lambda = lambda,
            device = identity,
        )
        push!(conditional_seqs, redesigned_sequence(g))
    end

    println("\n=== Generating $n_samples unconditional samples (steps=$steps) ===")
    for i in 1:n_samples
        print("  Sample $i/$n_samples ")
        x1 = make_design_target(template)
        g = design(model, x1; steps = steps, printseq = false, device = identity)
        push!(unconditional_seqs, redesigned_sequence(g))
        println()
    end

    return unconditional_seqs, conditional_seqs
end

end
