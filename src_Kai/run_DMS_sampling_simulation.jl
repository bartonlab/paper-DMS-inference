using DelimitedFiles
using Distributed
using StatsBase 
using Distributions
using Random
using Statistics
using Printf
using CSV
using LinearAlgebra
using DataFrames
rng = Random.MersenneTwister(1234);
include("./tools_DMS_sampling_simulation.jl")
fname_csv = ARGS[1] # "./output/simulation/init_sampling.csv"
fname_selection = ARGS[2] # "./out_simulation/selection_GT_Ntot-1000000_T-1-6-11_nR-3_sys.txt"
dir_out = ARGS[3] # "./out_simulation"
Ntot = parse(Int, ARGS[4]) # Size of total poulation where samples are taken from.
N = parse(Int, ARGS[5]) # 10_000
n_amplification = parse(Int, ARGS[6])

# Load data
csv_raw = CSV.read(fname_csv, DataFrame)
# Extract codon_list from the columns starting from the third column
codon_list = names(csv_raw)[3:end];
# Make a dictionary mapping codon to index. 
codon2num = Dict(s => i for (i, s) in enumerate(codon_list))

site_set = sort(unique(csv_raw.site))
L_eff = length(site_set);
q = length(codon_list); 
q_nuc = 3;

# Set selection coeficient
s_GT = vec(readdlm(fname_selection));
# Make sure the wild type codon has a fitness value of about 1.0 
seq_wt = [codon2num[x] for x in csv_raw.wildtype];
#s_GT[km.(1:L_eff, seq_wt, q)] .= 0.0; # -- ensure the selection pressure of the wildtype is zero. 

seq_wt_joint = join(seq_wt, ".")
f_wt = 1.0 + sum(s_GT[km.(1:L_eff, parse.(Int, split(seq_wt_joint, ".")), q)])
@printf("f_WT = %.2f\n", f_wt)

# Set initial (large) complete hapolotype population from the codon count tables. 
@time pop_init = get_initial_complete_population(q, L_eff, Ntot, csv_raw, seq_wt);


T = 10; # Number of generations.
nR = 3; # Number of replicates.
sampling_time = [1, 6, 11];

# ============================================================================
# Resample sequences N individuals from the total population. 
flag_uniform_sample=true;
@time pop_new = resample_population(pop_init, N, flag_uniform_sample)
# Reduce the population by removing individual wth no counts (never observed)
@time pop_new = reduce_population(q, L_eff, s_GT, [make_clone(s) for s in pop_new], seq_wt);

#### --- Increase the population about 20 times.  ###
pop_new[1].n=1
poisson_dist = Poisson(n_amplification)
for s in pop_new
    s.n = s.n * (rand(poisson_dist) + 1)
end;

N_evo_all_replicates, N_evo_NM_all_replicates = [], []
for i_r in 1:nR
    ### ======= Multinomial process ======= #
    # Start:=================================
    #Note: the population size is N, which is smaller than N_{tot}, the complete population size. 
    pop = [make_clone(s) for s in pop_new]
    # Initialize matrix for tracking population sizes over multiple cycles
    N_tot_evo = zeros(Int, length(pop), length(sampling_time) )
    N_tot_evo_NM = zeros(Int, length(pop), length(sampling_time) )
    
    num_pop_temporal = [pop[i].n for i in 1:length(pop)]
    r_set = get_r.(num_pop_temporal)
    r_set[r_set .<= 1e-3] .= 1e-3 # Prevent from error. In this case r is proportional to mean value, smaller one will be zero anyway.
    n_set_temp_NM = negative_multinomial_given_dispersion(num_pop_temporal, r_set)

    for i in 1:length(pop)
        N_tot_evo[i, 1] = pop[i].n
        N_tot_evo_NM[i, 1] = n_set_temp_NM[i]
    end

    # Sampling in multiple cycles
    for k in 2:(T+1)
        f_and_p = [s.n * s.f for s in pop]  # Compute product of population size and fitness
        f_and_p[f_and_p .<= 0] .= 0.0  # Set non-positive fitness to zero, as they shouldn't be observed
        prob_r = f_and_p / sum(f_and_p)  # Normalize to get probabilities
        @printf("Rep.=%d, gen.=%d, fitness=%.2e\n",  i_r, k, sum(f_and_p) )
        @time n_set_temp = rand(Multinomial(N * n_amplification, prob_r))  # Sample the next generation based on probabilities
        
        # ---- Adding noise though negative binomial distribution.
        for i in 1:length(pop)
            pop[i].n = n_set_temp[i]  # Update population size
        end
        # The following function should b
        # -- sampling --
        if(k ∈ sampling_time)
            k_eff = findfirst(sampling_time .== k)
            # NB sampling should be performed just before the sampling time.
            num_pop_temporal = copy(n_set_temp)
            num_pop_temporal[num_pop_temporal .< 0 ] .= 0.0
            r_set = get_r.(num_pop_temporal)
            r_set[r_set .<= 1e-3] .= 1e-3 # Prevent from error. In this case r is proportional to mean value, smaller one will be zero anyway.
            n_set_temp_NM = negative_multinomial_given_dispersion(num_pop_temporal, r_set)        
            for i in 1:length(pop)
                N_tot_evo[i, k_eff] = n_set_temp[i]  # Track evolution over cycles
                N_tot_evo_NM[i, k_eff] = n_set_temp_NM[i]  # Track evolution over cycles 
            end
        end        
    end;
    push!(N_evo_all_replicates, N_tot_evo)
    push!(N_evo_NM_all_replicates, N_tot_evo_NM)
    # =================================: End #
end

### Output to the result ### 
dir_make = @sprintf("N-%d_Ntot-%d_T-%s_nR-%d_sys", N, Ntot, @sprintf("%s", join(sampling_time, "-")), nR)

# Check if the directory exists, and create it if it does not
if !isdir(dir_out * dir_make)
    mkdir(dir_out * dir_make)
    mkdir(dir_out * dir_make * "/multinomial")
    mkdir(dir_out * dir_make * "/negative-multinomial")
    println("Directory created at: ", dir_out * dir_make)
else
    println("Directory already exists at: ", dir_out * dir_make)
end

fkey = @sprintf("N-%d_Ntot-%d_T-%s_nR-%d_sys", N, Ntot, @sprintf("%s", join(sampling_time, "-")), nR)
# --- Write true selection coefficient ---- #
df = DataFrame(
    position = [i for i in 1:L_eff for _ in 1:q],
    codon = vec(repeat(codon_list, L_eff,1)), 
    selection = s_GT)
CSV.write(@sprintf("%s/selection_%s.csv", dir_out * dir_make, fkey), df)

# --- Write csv files --- #
seq_set_temp = []
for n in 1:length(pop_new)
    codon_set_temp = [codon_list[i] for i in parse.(Int, split(pop_new[n].seq, "."))]
    seq_str = join([codon2aa[x] for x in codon_set_temp])
    push!(seq_set_temp, seq_str)
end;
true_false_set = [s.seq == seq_wt_joint for s in pop_new]
fitness_set = [s.f for s in pop_new]
# ---- Multinomial Processes ---#
df = DataFrame(
    sequence=seq_set_temp, 
    wildtype=true_false_set,
    fitness=fitness_set)
for i_r in 1:nR
    for i_t in 1:length(sampling_time)
        df[!, Symbol( @sprintf("%d_c_%d", i_r, sampling_time[i_t] ) )] = N_evo_all_replicates[i_r][:, i_t]
    end
end
CSV.write(@sprintf("%s/count_%s_multinomial.csv", dir_out * dir_make, fkey), df)

# ---- Negative Multinomial Processes ---#
df = DataFrame(
    sequence=seq_set_temp, 
    wildtype=true_false_set,
    fitness=fitness_set)
for i_r in 1:nR
    for i_t in 1:length(sampling_time)
        df[!, Symbol( @sprintf("%d_c_%d", i_r, sampling_time[i_t] ) )] = N_evo_NM_all_replicates[i_r][:, i_t]
    end
end
CSV.write(@sprintf("%s/count_%s_negative_multinomial.csv", dir_out * dir_make, fkey), df);

#### Making files for preference  
q_aa = length(amino_acids)
#fname_multinomial = "./out_simulation/N-10000_Ntot-1000000_T-10_nR-10/count_N-10000_Ntot-1000000_T-10_nR-10_multinomial.csv"
fname_multinomial = @sprintf("%s/%s/count_%s_multinomial.csv", dir_out, fkey, fkey)
csv_count_raw_multi = CSV.read(fname_multinomial, DataFrame);

#fname_neg_multi = "./out_simulation/N-10000_Ntot-1000000_T-10_nR-10_v2/count_N-10000_Ntot-1000000_T-10_nR-10_negative_multinomial.csv"
fname_neg_multi = @sprintf("%s/%s/count_%s_negative_multinomial.csv", dir_out, fkey, fkey)
csv_count_raw_neg_multi = CSV.read(fname_neg_multi, DataFrame);

# --- Get frequency tables for all replicates and rounds.
freq_full_set_mult = get_freq_tables_from_csv(nR, sampling_time, q_aa, L_eff, csv_count_raw_multi);
freq_full_set_neg_mult = get_freq_tables_from_csv(nR, sampling_time, q_aa, L_eff, csv_count_raw_neg_multi);

# --- Get tables of enrichment that compared initial round and a temporal round t. 
(enrichment_mult, enrichment_in_log_regression_mult, enrichment_log_reg_av_mult) = get_enrichment_from_freq_table(nR, sampling_time, q_aa, L_eff, freq_full_set_mult )
(enrichment_neg_mult, enrichment_in_log_regression_neg_mult, enrichment_log_reg_av_neg_mult) = get_enrichment_from_freq_table(nR, sampling_time, q_aa, L_eff, freq_full_set_neg_mult );
enrichment_mult_av = [mean(enrichment_mult[:, end, i]) for i in 1:length(enrichment_mult[1, 1, :]) ]
enrichment_neg_mult_av = [mean(enrichment_neg_mult[:, end, i]) for i in 1:length(enrichment_neg_mult[1, 1, :]) ];

# ---- Get preference values 
preference_mult = get_preferences_from_enrichment(nR, q_aa, L_eff, enrichment_mult) 
preference_neg_mult = get_preferences_from_enrichment(nR, q_aa, L_eff, enrichment_neg_mult)
preference_mult_av = [mean(preference_mult[:, i]) for i in 1:(q_aa * L_eff)]
preference_neg_mult_av = [mean(preference_neg_mult[:, i]) for i in 1:(q_aa * L_eff)];
# --------------------------------------------------------- #

# --- Write results to a csv file 
# multinomial process
df = DataFrame(
    position=[i for i in 1:L_eff for _ in 1:q_aa], 
    AA_i=[x for _ in 1:L_eff for x in amino_acids])
df[!,  Symbol( "enrich_av" ) ] = enrichment_mult_av;
df[!,  Symbol( "preference_av" ) ] = preference_mult_av;
df[!,  Symbol( "enrich_regress_av" ) ] = enrichment_log_reg_av_mult[2, :];
for i_R in 1:nR
    df[!,  Symbol( @sprintf("enrich_Rep%d", i_R) ) ] = enrichment_mult[i_R, end, :]
end
for i_R in 1:nR
    df[!,  Symbol( @sprintf("preference_Rep%d", i_R) ) ] = preference_mult[i_R, :]
end
for i_R in 1:nR
    df[!,  Symbol( @sprintf("enrich_regress_Rep%d", i_R) ) ] = enrichment_in_log_regression_mult[i_R, :]
end
CSV.write( @sprintf("%s%s/multinomial/preference_multinomial.csv", dir_out, fkey) , df);

# negative multinomial process
df = DataFrame(
    position=[i for i in 1:L_eff for _ in 1:q_aa], 
    AA_i=[x for _ in 1:L_eff for x in amino_acids])
df[!,  Symbol( "enrich_av" ) ] = enrichment_neg_mult_av;
df[!,  Symbol( "preference_av" ) ] = preference_neg_mult_av;
df[!,  Symbol( "enrich_regress_av" ) ] = enrichment_log_reg_av_neg_mult[2, :];
for i_R in 1:nR
    df[!,  Symbol( @sprintf("enrich_Rep%d", i_R) ) ] = enrichment_neg_mult[i_R, end, :]
end
df[!,  Symbol( "enrich_av" ) ] = enrichment_neg_mult_av;
for i_R in 1:nR
    df[!,  Symbol( @sprintf("preference_Rep%d", i_R) ) ] = preference_neg_mult[i_R, :]
end
for i_R in 1:nR
    df[!,  Symbol( @sprintf("enrich_regress_Rep%d", i_R) ) ] = enrichment_in_log_regression_neg_mult[i_R, :]
end
CSV.write(@sprintf("%s%s/negative-multinomial/preference_negative_multinomial.csv", dir_out, fkey), df);
