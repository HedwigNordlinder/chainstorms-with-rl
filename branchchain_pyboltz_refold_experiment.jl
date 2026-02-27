#!/usr/bin/env julia
#=
BranchChain + PyBoltz DTS* binder design
=======================================
Design exactly one chain (binder) in a template complex using BranchChain DTS*,
and score each rollout endpoint using PyBoltz ipSAE.

Expected input:
- A multi-chain template PDB where one chain is the fixed target and one chain
  is the binder scaffold to redesign, OR a single-chain target PDB.
- For single-chain inputs, a synthetic binder scaffold chain is automatically
  created by copying target backbone fragments with a translation offset.

Configuration is controlled through environment variables (see `main()`).
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "BranchChain.jl"))
if isnothing(Base.find_package("PyBoltz"))
    Pkg.develop(path = joinpath(@__DIR__, "PyBoltz.jl"))
end

using BranchChain
using DLProteinFormats
using ProteinChains: readpdb, Atom, ProteinChain, ProteinStructure

const DEFAULT_TEMPLATE_PATH = joinpath(@__DIR__, "15-pgdh.pdb")
const DEFAULT_CHECKPOINT = "branchchain_tune1.jld"
const DEFAULT_BRANCHING_POINTS = [0.0, 0.05, 0.15, 0.4, 0.7]
const REWARD_FAILURE_PENALTY = -1.0e6
const PYBOLTZ_LOADED = Ref(false)

parse_int_env(name::AbstractString, default::Integer) = parse(Int, get(ENV, name, string(default)))
parse_float_env(name::AbstractString, default::Real) = parse(Float64, get(ENV, name, string(default)))

function parse_branching_points_env(name::AbstractString, default::Vector{Float64})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    vals = [parse(Float64, strip(x)) for x in split(raw, ",") if !isempty(strip(x))]
    isempty(vals) && return default
    return vals
end

function maybe_env_string(name::AbstractString)
    raw = strip(get(ENV, name, ""))
    return isempty(raw) ? nothing : raw
end

function get_chain_ids(template)
    return [string(chain.id) for chain in template]
end

function choose_unused_chain_id(existing_ids::Vector{String}; preferred::String = "B")
    preferred in existing_ids || return preferred
    for c in 'A':'Z'
        id = string(c)
        id in existing_ids || return id
    end
    return "X$(length(existing_ids) + 1)"
end

function pick_chain_ids(template; target_chain_id = nothing, binder_chain_id = nothing)
    chain_ids = get_chain_ids(template)
    length(chain_ids) >= 2 || error("Template must contain at least 2 chains (target + binder scaffold). Found: $(join(chain_ids, ", ")).")

    binder = isnothing(binder_chain_id) ? chain_ids[end] : binder_chain_id
    binder in chain_ids || error("Binder chain '$binder' not found in template. Available: $(join(chain_ids, ", ")).")

    if isnothing(target_chain_id)
        candidates = filter(!=(binder), chain_ids)
        isempty(candidates) && error("No target chain available after selecting binder '$binder'.")
        target = first(candidates)
    else
        target = target_chain_id
    end

    target in chain_ids || error("Target chain '$target' not found in template. Available: $(join(chain_ids, ", ")).")
    target != binder || error("Target chain and binder chain must be different, got '$target'.")

    return target, binder
end

function make_synthetic_binder_chain(target_chain;
    binder_chain_id::String,
    binder_length::Int = 48,
    binder_offset::Float64 = 12.0,
)
    target_length = length(target_chain.sequence)
    target_length > 0 || error("Cannot build synthetic binder from an empty target chain.")
    L = max(1, binder_length)

    binder_atoms = Vector{Vector{typeof(target_chain.atoms[1][1])}}()
    seqbuf = IOBuffer()
    for i in 1:L
        src_idx = mod1(i, target_length)
        src_atoms = target_chain.atoms[src_idx]
        new_atoms = [Atom(a.name, a.number, a.x + binder_offset, a.y, a.z) for a in src_atoms]
        push!(binder_atoms, new_atoms)
        print(seqbuf, target_chain.sequence[src_idx])
    end
    binder_seq = String(take!(seqbuf))
    return ProteinChain(binder_chain_id, binder_atoms, binder_seq, collect(1:L))
end

function prepare_template_for_binder_design(base_template;
    target_chain_id = nothing,
    binder_chain_id = nothing,
    binder_length::Int = 48,
    binder_offset::Float64 = 12.0,
)
    chain_ids = get_chain_ids(base_template)
    if length(chain_ids) == 1
        target = isnothing(target_chain_id) ? chain_ids[1] : target_chain_id
        target in chain_ids || error("Target chain '$target' not found in single-chain template.")

        target_chain = only(base_template)
        requested_binder = isnothing(binder_chain_id) ? "B" : binder_chain_id
        binder = choose_unused_chain_id(chain_ids; preferred = requested_binder)
        synthetic_binder = make_synthetic_binder_chain(
            target_chain;
            binder_chain_id = binder,
            binder_length = binder_length,
            binder_offset = binder_offset,
        )
        templ_name = "$(base_template.name)_synthetic_binder"
        template = ProteinStructure(templ_name, [target_chain, synthetic_binder])
        return template, target, binder, true
    end

    target, binder = pick_chain_ids(base_template; target_chain_id, binder_chain_id)
    return base_template, target, binder, false
end

function chain_sequence_by_id(template, chain_id::AbstractString)
    for chain in template
        if string(chain.id) == chain_id
            return chain.sequence
        end
    end
    error("Chain '$chain_id' not found.")
end

function make_binder_design_target(template, binder_chain_id::AbstractString)
    chain_ids = get_chain_ids(template)
    binder_idx = findfirst(==(binder_chain_id), chain_ids)
    isnothing(binder_idx) && error("Binder chain '$binder_chain_id' not found in template.")

    binder_seq = chain_sequence_by_id(template, binder_chain_id)
    exclude_flatchain_nums = setdiff(collect(1:length(chain_ids)), [binder_idx])
    x1 = X1_from_pdb(template, [binder_seq]; exclude_flatchain_nums = exclude_flatchain_nums)
    return x1
end

function redesigned_binder_sequence(state)
    aas = Int.(BranchChain.tensor(state.state[3]))
    mask = Bool.(state.flowmask)
    n = min(length(aas), length(mask))
    n == 0 && return ""

    aa_chars = collect(DLProteinFormats.ints_to_aa(aas[1:n]))
    out = IOBuffer()
    for i in 1:n
        if mask[i]
            aa = aa_chars[i]
            aa == 'X' && continue
            print(out, aa)
        end
    end
    return String(take!(out))
end

function ensure_pyboltz_loaded()
    if !PYBOLTZ_LOADED[]
        @eval using PyBoltz
        PYBOLTZ_LOADED[] = true
    end
end

function make_pyboltz_reward(target_sequence::String;
    accelerator::String = "cpu",
    sampling_steps::Int = 30,
    recycling_steps::Int = 3,
    diffusion_samples::Int = 1,
    base_seed::Int = 0,
)
    score_cache = Dict{String, Float64}()
    eval_counter = Ref(0)

    function reward_fn(state)
        binder_sequence = redesigned_binder_sequence(state)
        isempty(binder_sequence) && return REWARD_FAILURE_PENALTY

        if haskey(score_cache, binder_sequence)
            return score_cache[binder_sequence]
        end

        eval_counter[] += 1
        score = try
            ensure_pyboltz_loaded()
            input = PyBoltz.Schema.BoltzInput(
                sequences = [
                    PyBoltz.Schema.protein(id = "A", sequence = target_sequence, msa = "empty"),
                    PyBoltz.Schema.protein(id = "B", sequence = binder_sequence, msa = "empty"),
                ],
                properties = [PyBoltz.Schema.affinity(binder = "B")],
            )
            Float64(PyBoltz.IPSAE.predict_ipsae(
                input;
                accelerator = accelerator,
                sampling_steps = sampling_steps,
                recycling_steps = recycling_steps,
                diffusion_samples = diffusion_samples,
                seed = base_seed + eval_counter[] - 1,
            ))
        catch err
            @warn "PyBoltz scoring failed, assigning penalty." err binder_length = length(binder_sequence)
            REWARD_FAILURE_PENALTY
        end

        score_cache[binder_sequence] = score
        println("    PyBoltz eval $(eval_counter[]): len=$(length(binder_sequence)) ipSAE=$(round(score, digits=4))")
        return score
    end

    return reward_fn, score_cache
end

function run_dts_pyboltz_binder_design(;
    template_path::String,
    checkpoint::String,
    target_chain_id::Union{Nothing, String},
    binder_chain_id::Union{Nothing, String},
    binder_length::Int,
    binder_offset::Float64,
    out_pdb::String,
    steps::Int,
    n_iterations::Int,
    max_children::Int,
    branching_points::Vector{Float64},
    c_uct::Float64,
    lambda::Float64,
    pb_accelerator::String,
    pb_sampling_steps::Int,
    pb_recycling_steps::Int,
    pb_diffusion_samples::Int,
    pb_seed::Int,
)
    isfile(template_path) || error("Template PDB not found: $template_path")
    mkpath(dirname(out_pdb))

    base_template = readpdb(template_path)
    template, target_chain_id, binder_chain_id, synthetic_binder = prepare_template_for_binder_design(
        base_template;
        target_chain_id,
        binder_chain_id,
        binder_length,
        binder_offset,
    )
    target_sequence = chain_sequence_by_id(template, target_chain_id)

    println("Template: $template_path")
    println("Target chain (fixed): $target_chain_id (len=$(length(target_sequence)))")
    println("Binder chain (redesigned): $binder_chain_id")
    if synthetic_binder
        println("Synthetic binder scaffold: yes (length=$binder_length, offset_x=$binder_offset A)")
    else
        println("Synthetic binder scaffold: no (using provided binder scaffold chain)")
    end

    x1 = make_binder_design_target(template, binder_chain_id)
    model = load_model(checkpoint)

    reward_fn, score_cache = make_pyboltz_reward(
        target_sequence;
        accelerator = pb_accelerator,
        sampling_steps = pb_sampling_steps,
        recycling_steps = pb_recycling_steps,
        diffusion_samples = pb_diffusion_samples,
        base_seed = pb_seed,
    )

    println("\n=== Running DTS* with PyBoltz reward ===")
    designed_state = design_treegen(
        model, x1;
        reward = reward_fn,
        steps = steps,
        n_iterations = n_iterations,
        max_children = max_children,
        branching_points = branching_points,
        c_uct = c_uct,
        lambda = lambda,
        device = identity,
        path = out_pdb,
    )

    best_binder_sequence = redesigned_binder_sequence(designed_state)
    best_score = get(score_cache, best_binder_sequence) do
        reward_fn(designed_state)
    end

    results_path = replace(out_pdb, r"\.pdb$" => "_summary.txt")
    open(results_path, "w") do io
        println(io, "template_path=$template_path")
        println(io, "target_chain_id=$target_chain_id")
        println(io, "binder_chain_id=$binder_chain_id")
        println(io, "best_ipSAE=$best_score")
        println(io, "best_binder_length=$(length(best_binder_sequence))")
        println(io, "best_binder_sequence=$best_binder_sequence")
        println(io, "unique_pyboltz_evals=$(length(score_cache))")
    end

    println("\n=== Finished ===")
    println("Best ipSAE: $(round(best_score, digits=4))")
    println("Best binder length: $(length(best_binder_sequence))")
    println("Best binder sequence: $best_binder_sequence")
    println("Saved structure: $out_pdb")
    println("Saved summary:   $results_path")
    println("Unique PyBoltz sequence evals: $(length(score_cache))")
end

function main()
    template_path = get(ENV, "BC_TEMPLATE_PATH", DEFAULT_TEMPLATE_PATH)
    checkpoint = get(ENV, "BC_CHECKPOINT", DEFAULT_CHECKPOINT)
    target_chain_id = maybe_env_string("BC_TARGET_CHAIN")
    binder_chain_id = maybe_env_string("BC_BINDER_CHAIN")
    binder_length = parse_int_env("BC_BINDER_LENGTH", 48)
    binder_offset = parse_float_env("BC_BINDER_OFFSET", 12.0)
    out_pdb = get(ENV, "BC_OUT_PDB", joinpath(@__DIR__, "branchchain_pyboltz_best_binder.pdb"))

    steps = parse_int_env("BC_STEPS", 30)
    n_iterations = parse_int_env("BC_TREE_ITERS", 20)
    max_children = parse_int_env("BC_MAX_CHILDREN", 3)
    branching_points = parse_branching_points_env("BC_BRANCHING_POINTS", DEFAULT_BRANCHING_POINTS)
    c_uct = parse_float_env("BC_C_UCT", 1.0)
    lambda = parse_float_env("BC_LAMBDA", 10.0)

    pb_accelerator = get(ENV, "PB_ACCELERATOR", "cpu")
    pb_sampling_steps = parse_int_env("PB_SAMPLING_STEPS", 30)
    pb_recycling_steps = parse_int_env("PB_RECYCLING_STEPS", 3)
    pb_diffusion_samples = parse_int_env("PB_DIFFUSION_SAMPLES", 1)
    pb_seed = parse_int_env("PB_SEED", 0)
    target_display = something(target_chain_id, "(auto)")
    binder_display = something(binder_chain_id, "(auto)")

    println("BranchChain + PyBoltz binder design (DTS*)")
    println("  BC_TEMPLATE_PATH=$template_path")
    println("  BC_TARGET_CHAIN=$target_display")
    println("  BC_BINDER_CHAIN=$binder_display")
    println("  BC_BINDER_LENGTH=$binder_length, BC_BINDER_OFFSET=$binder_offset")
    println("  BC_OUT_PDB=$out_pdb")
    println("  BC_STEPS=$steps, BC_TREE_ITERS=$n_iterations, BC_MAX_CHILDREN=$max_children")
    println("  BC_BRANCHING_POINTS=$(join(branching_points, ","))")
    println("  PB_ACCELERATOR=$pb_accelerator, PB_SAMPLING_STEPS=$pb_sampling_steps, PB_RECYCLING_STEPS=$pb_recycling_steps, PB_DIFFUSION_SAMPLES=$pb_diffusion_samples")

    run_dts_pyboltz_binder_design(;
        template_path,
        checkpoint,
        target_chain_id,
        binder_chain_id,
        binder_length,
        binder_offset,
        out_pdb,
        steps,
        n_iterations,
        max_children,
        branching_points,
        c_uct,
        lambda,
        pb_accelerator,
        pb_sampling_steps,
        pb_recycling_steps,
        pb_diffusion_samples,
        pb_seed,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
