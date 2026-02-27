# Issues Review

## Findings (ordered by severity)

1. **[P1] Soft Bellman backup is implemented as log-mean-exp, not log-sum-exp.**  
   In the paper’s DTS/DTS* algorithm, backup is `v = (1/λ) log Σ exp(λ v_child)`. The code subtracts `log(n_children)`, changing it to log-mean-exp and removing subtree-mass effects that DTS* relies on.  
   File: `ChainStorm.jl/src/treegen.jl:167`  
   File: `ChainStorm.jl/src/treegen.jl:168`

2. **[P1] Progressive widening is missing; branching is hard-capped.**  
   The paper uses `B(x)=C*N(x)^α`; the implementation uses fixed `max_children`. This breaks the paper’s scaling/asymptotic behavior and eventually prevents further structural growth per node.  
   File: `ChainStorm.jl/src/treegen.jl:65`  
   File: `ChainStorm.jl/src/treegen.jl:123`  
   File: `ChainStorm.jl/src/treegen.jl:189`

3. **[P2] Final return can discard the best discovered terminal sample.**  
   `best_leaf` can end at a nonterminal frontier node; then `flow_treegen` does a fresh stochastic rollout and returns that new sample. This can be worse than the best sample already observed in-tree.  
   File: `ChainStorm.jl/src/treegen.jl:176`  
   File: `ChainStorm.jl/src/treegen.jl:247`

4. **[P2] Branch schedule inputs are not validated.**  
   No checks that `branching_points` are sorted, within `[0,1]`, unique, and starting at `0.0`. Bad inputs can silently produce malformed/no-op segments and incorrect denoising coverage.  
   File: `ChainStorm.jl/src/treegen.jl:69`  
   File: `ChainStorm.jl/src/treegen.jl:205`

5. **[P2] Reward experiments ignore chain boundaries, so motifs can match across chain junctions.**  
   `extract_sequences` converts the full residue axis to one string, even for multi-chain batches (`[40,40]`). This makes motif rewards inconsistent with per-chain biology claims.  
   File: `fxf_experiment.jl:23`  
   File: `walkera_experiment.jl:23`  
   File: `cysteine_experiment.jl:23`

6. **[P2] No tests cover `flow_treegen` behavior.**  
   Current tests only validate loss outputs for a tiny model; tree search logic has no correctness tests (selection/expansion/backup invariants).  
   File: `ChainStorm.jl/test/runtests.jl:4`

7. **[P3] Deprecated API usage warnings in model forward pass.**  
   `Flux.Zygote.@ignore` is deprecated (warnings during `Pkg.test`).  
   File: `ChainStorm.jl/src/model.jl:28`  
   File: `ChainStorm.jl/src/model.jl:29`  
   File: `ChainStorm.jl/src/model.jl:31`

## Verification run

- `flow_treegen` smoke test with a small `ChainStormV1(16,2,2)` model executed successfully.
- `Pkg.test()` in `ChainStorm.jl` passed (`3/3`), with deprecation warnings noted above.
