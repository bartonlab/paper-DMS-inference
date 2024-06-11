km(i,a,q) = (i-1)*q + a

# Define a dictionary, similar to Python's dict, representing the DNA codon table
aa2codon = Dict(
    'A' => ["GCT", "GCC", "GCA", "GCG"],
    'R' => ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    'N' => ["AAT", "AAC"],
    'D' => ["GAT", "GAC"],
    'C' => ["TGT", "TGC"],
    'E' => ["GAA", "GAG"],
    'Q' => ["CAA", "CAG"],
    'G' => ["GGT", "GGC", "GGA", "GGG"],
    'H' => ["CAT", "CAC"],
    'I' => ["ATT", "ATC", "ATA"],
    'L' => ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    'K' => ["AAA", "AAG"],
    'M' => ["ATG"],
    'F' => ["TTT", "TTC"],
    'P' => ["CCT", "CCC", "CCA", "CCG"],
    'S' => ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    'T' => ["ACT", "ACC", "ACA", "ACG"],
    'W' => ["TGG"],
    'Y' => ["TAT", "TAC"],
    'V' => ["GTT", "GTC", "GTA", "GTG"],
    '*' => ["TAA", "TGA", "TAG"]
    #'-' => ["---"]
);
codon2aa = Dict{String, Char}()

# Populate the new dictionary
for (aa, codons) in aa2codon
    for codon in codons
        codon2aa[codon] = aa
    end
end

amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*']
aa2num = Dict(amino_acids[i] => i for i in 1:length(amino_acids))
number_to_amino_acid = Dict(value => key for (key, value) in aa2num)

mutable struct Species
    id::Int
    n::Int
    f::Float32
    seq::String
end

make_clone(s) = Species(s.id, s.n, s.f, s.seq);

function get_s_GT(q, L_eff, codon_list, σ=0.1/sqrt(3))
    s_GT = zeros(q*L_eff)#0.1 * randn(q * L_eff);
    for i in 1:L_eff
        aa_rand = σ*randn(aa2codon.count);
        aa_list_tmp = [codon2aa[x] for x in codon_list]
        idx_map_codon2aa = [aa2num[x] for x in aa_list_tmp]
        codon_rand = [aa_rand[i] for i in idx_map_codon2aa]
        s_GT[km.(i, 1:q, q)] = codon_rand
    end;
    return s_GT
end;

function negative_multinomial(X0, P, N_itr=10000000)
    # Total successes needed
    total_successes = zeros(Int, length(P))
    count_samples = 0
    # Continue sampling until the cumulative successes in one category reach `r`
    while sum(total_successes) < X0 * N_itr
        # Sample from a standard multinomial distribution
        sample = rand(Multinomial(1, P))
        total_successes += sample
        count_samples += 1

        # Check if we have enough successes
        if any(total_successes .>= X0)
            break
        end
    end

    return total_successes, count_samples
end

get_r(λ) = 0.8*λ^0.69 ; #  ACIDE paper # https://www.nature.com/articles/s41467-023-43967-9 
function negative_multinomial_given_dispersion(mean_set, r_set)
    #r_set = get_r.(mean_set)
    p_values = r_set ./ (r_set + mean_set)
    len_p = length(p_values)
    sample_NM = zeros(Int, len_p)
    for i in 1:len_p
        sample_NM[i] = rand(NegativeBinomial(r_set[i], p_values[i]))
    end
    return sample_NM
end

# Get number of stoping iteration give prob
get_X0(N, P) = Int( floor( N / sum( P / maximum(P) ) ) );


function get_initial_complete_population(q, L_eff, Ntot, csv_raw, seq_wt, α=0.0) # previously α=0.0005
    # Making site independent model
    individual_site_freq = zeros(L_eff, q);  # corrected spelling of 'individual_site_freq'
    for i in 1:L_eff
        count_temp = [x for x in csv_raw[i, 3:end]]  # this assumes csv_raw is correctly loaded and indexed
        prob_temp = (1.0-α) * count_temp / sum(count_temp) .+ α/q  # correctly calculating probabilities
        individual_site_freq[i, :] = copy(prob_temp)  # corrected spelling of 'individual_site_freq'
    end;

    # Sampling sequences
    seq_init = zeros(Int, Ntot, L_eff);  # corrected spelling of 'seq_init' and other identifiers
    for i in 1:L_eff
        prob_r = individual_site_freq[i, :];  # corrected spelling, accessing the computed frequencies
        n_mult = rand(Multinomial(Ntot, prob_r));  # sampling of number of N sequences
        n_mult_nnz = n_mult[n_mult .> 0]  # the number of samples only for non-zero values
        a_mult_nnz = findall(n_mult .> 0);  # indices of the states with non-zero values (corrected to use findall)

        # Distributing the sampled q-states to N sequences.
        n_set_temp = vcat([repeat([a_mult_nnz[k]], n_mult_nnz[k]) for k in 1:length(a_mult_nnz)]...)
        n_set_shuffled = shuffle(n_set_temp)
        seq_init[:, i] = n_set_shuffled
    end

    # Reformatting sequence vector to a string.
    seq_init_join = []
    for n in 1:Ntot 
        push!( seq_init_join, join(seq_init[n, :], ".") )
    end  
    # Store sequences in a list
    pop = []
    seq_init_join_unique = unique(seq_init_join)
    # Since it is a high-dimensional space, a sequence that we observe multiple times is almost always WT.
    seq_wt_joint = join(seq_wt, ".")
    n = count(x -> x == seq_wt_joint, seq_init_join)
    f_wt = 1.0 + sum(s_GT[km.(1:L_eff, seq_wt, q)])  # Assuming s_GT and km functions are defined elsewhere
    push!(pop, Species(0, n, f_wt, seq_wt_joint));
    idx = [s != seq_wt_joint for s in seq_init_join_unique]
    seq_init_join_unique = copy(seq_init_join_unique[idx])
    # Even though sequences can be identical, we don't distinguish at this moment.
    for i in 1:length(seq_init_join_unique)
        seq_temp = seq_init_join_unique[i]
        seq_num = parse.(Int, split(seq_temp, "."))
        push!(pop, Species(i, 1, 1, seq_temp));
    end;
    return pop
end;

# Resampling using just low count
function resample_population(pop, N, flag_uniform_sample)
    f_and_p = [s.n for s in pop]
    
    if(flag_uniform_sample) # Sample individuals uniformly randomly. 
        f_and_p = [1.0 for s in pop]
    end;
    
    prob_r = f_and_p / sum(f_and_p)
    n_set_temp = rand(Multinomial(N, prob_r))
    
    pop_out = []
    for i_n in 1:length(n_set_temp)
        n = n_set_temp[i_n]
        if(n>0)
            push!(pop_out, make_clone(pop[i_n]) )
            pop_out[end].n = n  # Corrected to set the new population size
        end
    end
    return pop_out
end

# Reduce the number of individuals; keep only observed individuals
function reduce_population(q, L_eff, s_GT, pop, seq_wt)
    pop = copy(pop[getindex.([s.n for s in pop] .> 0)])
    seq_init_join = [s.seq for s in pop]
    seq_init_join_unique = unique(seq_init_join)
    
    pop_new = []
    seq_wt_joint = join(seq_wt, ".")
    n = sum([x.n for x in pop] .* (seq_init_join .== seq_wt_joint))
    f_wt = 1.0 + sum(s_GT[km.(1:L_eff, parse.(Int, split(seq_wt_joint, ".")), q)])
    push!(pop_new, Species(0, n, f_wt, seq_wt_joint))

    idx = [s != seq_wt_joint for s in seq_init_join_unique]
    seq_init_join_unique = seq_init_join_unique[idx]

    for i in 1:length(seq_init_join_unique)
        seq_temp = seq_init_join_unique[i]
        seq_num = parse.(Int, split(seq_temp, "."))
        f_temp = 1.0 + sum(s_GT[km.(1:L_eff, seq_num, q)])
        push!(pop_new, Species(i, 1, f_temp, seq_temp))
    end
    
    return pop_new
end;
    
function get_freq_tables_from_csv(nR, sampling_time, q_aa, L_eff, csv_count_raw_in)
    
    # -- get total number of observed genotypes-- #
    total_count_set = zeros(nR, length(sampling_time) )
    for i_R in 1:nR
        for i_T in 1:length(sampling_time)
            n_count_tot = sum(csv_count_raw_in[:, Symbol( @sprintf("%d_c_%d", i_R, sampling_time[i_T] ) ) ])
            total_count_set[i_R, i_T] = n_count_tot
        end
    end;
    
    # -- get total number of observed genotypes-- #
    freq_full_set = zeros(nR, length(sampling_time), q_aa * L_eff);
    for n in 1:length(csv_count_raw_in[:, 1])
        aa_num_set_temp = [aa2num[x[1]] for x in split(csv_count_raw_in.sequence[n], "")]
        for i_R in 1:nR
            for i_T in 1:length(sampling_time)
                n_count_in = csv_count_raw_in[n, Symbol( @sprintf("%d_c_%d", i_R, sampling_time[i_T]) ) ]
                freq_in = n_count_in / (total_count_set[i_R, i_T])
                freq_full_set[i_R, i_T, km.(1:L_eff, aa_num_set_temp, q_aa)] .+= freq_in
            end
        end
    end
    return freq_full_set
end;

function get_enrichment_from_freq_table(nR, sampling_time, q_aa, L_eff, freq_full_set_in, pseudo_count=1e-5, γ=1e-3)
    nT = length(sampling_time)
    enrichment_in = zeros(nR, nT-1, q_aa * L_eff)
    enrichment_in_log_regression = zeros(nR, q_aa * L_eff)
    enrichment_log_reg_av = zeros(2, q_aa * L_eff) # enrichment_log_reg_av[2, :] should be the same as the Enrich2 method.
    σ2_r_set = zeros(nR, q_aa * L_eff)

    t_mat = [ones(nT) sampling_time];
    tt_mat = t_mat' * t_mat
    @show size(tt_mat)
    A_denom = tt_mat + γ*I
    #A_denom[diagind(A_denom)] += γ
    inv_A_denom = inv(A_denom)
    @show size(inv_A_denom)
    
    for i_R in 1:nR
        temp_regress = zeros(nT, q_aa * L_eff)
        temp_regress[1, :] = log.(freq_full_set_in[i_R, 1, :] .+ pseudo_count)
        
        for i_T in 2:nT
            enrichment_in[i_R, i_T-1, :] = freq_full_set_in[i_R, i_T, :] ./ (freq_full_set_in[i_R, 1, :] .+ pseudo_count)
            temp_regress[i_T, :] = log.(freq_full_set_in[i_R, i_T, :] .+ pseudo_count)
        end
        
        # -- Regression of slope: α = ∑_k t_k e_k /  ∑_k t_k^2
        for i in 1:(q_aa * L_eff)
            β_iR_i = inv_A_denom * (t_mat' * temp_regress[:, i])
            enrichment_in_log_regression[i_R, i] = β_iR_i[2] # this gives slope 
            # Deviation between the regressed model and data at each replicate
            σ2_r_set[i_R, i] = sum( (t_mat * β_iR_i .- temp_regress[:, i]) .^ 2 ) / (nT - 1)
        end
    end;
    
    # -- Enrich2 integrate individually estimated regresson values as follows: 
    for i in 1:(q_aa * L_eff)
        β_al_av = mean(enrichment_in_log_regression[:, i]) # alithmetic average individually regressed value over replicates
        enrichment_log_reg_av[1, i] = β_al_av

        # Deviation of models among replicates
        σ2_s = var(enrichment_in_log_regression[:, i]) 

        # Deviation between the regressed model and data at each replicate
        σ2_r = σ2_r_set[:, i]

        # Get averaged log ratio regression using the Enrich2 scheme.
        inv_σ = 1.0 ./ ( σ2_s .+  σ2_r)
        β_Enrich2 = sum( enrichment_in_log_regression[:, i] .* inv_σ ) / sum(inv_σ)
        enrichment_log_reg_av[2, i] = β_Enrich2
    end
    
    return (enrichment_in, enrichment_in_log_regression, enrichment_log_reg_av)
end;

function get_preferences_from_enrichment(nR, q_aa, L_eff, enrichment)
    preference_out = zeros(nR, q_aa * L_eff)
    for i_R in 1:nR
        for i in 1:L_eff
            enrichment_i_iR = copy(enrichment[i_R, end, km.(i, 1:q_aa, q_aa)])
            preference_out[i_R, km.(i, 1:q_aa, q_aa)] = enrichment_i_iR / sum(enrichment_i_iR)
        end
    end
    return preference_out
end 
