using Statistics
#using StatsBase
using DelimitedFiles
using Distributions
using Random
using LinearAlgebra
using Printf
rng = Random.MersenneTwister(1234);
using DataFrames
using CSV

include("./tools_for_selection_inference.jl")

# ================= ARGUMNT =====================#
dir_in = ARGS[1]
dir_out = ARGS[2]
file_csv_in = ARGS[3] 
freq_th = parse(Float64, ARGS[4])
flag_out = parse(Bool, ARGS[5])
flag_load_cov_Dx = parse(Bool, ARGS[6])
dir_load = ARGS[7]
ref_seq = ARGS[8] 
end_exp = parse(Int, ARGS[9]) # =4
num_values = parse(Int, ARGS[10]) # = 20
#N_pop_tot = parse(Float64, ARGS[9])
#fname_reg = ARGS[10]
# ==============================================#
L, q = length(split(ref_seq, "")), 22
qL = q*L; 
x_rank = qL

@time csv_raw = DataFrame(CSV.File(dir_in * file_csv_in));
(ids_replicate, ids_rounds) = get_replication_round_ids(names(csv_raw));
γ_set = get_reg_set(ids_replicate, ids_rounds, csv_raw)
#inv_N_pop_tot = 1.0 / N_pop_tot;
#reg_set_without_scale = [x for x in readdlm(fname_reg)]
#γ_set = reg_set_without_scale * inv_N_pop_tot
#γ_set = [1e2, 1e3, 1e4] * inv_N_pop_tot
#γ_set = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7] * inv_N_pop_tot
#γ_set = ( 1e2 .+ [-50.0, -30.0, -20.0, -10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0 ]) * inv_N_pop_tot # This is temporal treatment to reproduce the past results. 
#  Process Start : ============================================== #

Δx_set, x_set, icov_set, iΔxΔxT_set = [], [], [], [] 
idx_detecable_i_a_set, idx_detecable_ij_ab_set = [], []
Δx = zeros(x_rank)
if(flag_load_cov_Dx)
    csv_raw = [] # free memory
    @printf("Start loading the integrated covariance and the correction in the integrated covariance.\n")
    @time (icov_set, iΔxΔxT_set, Δx_set) = load_icov_iΔxΔxT(dir_load, ids_replicate)
    @printf("Done.\n")
    # Get the detectable sites and states.
    for n_rep in 1:length(ids_replicate)  
        global Δx
        Δx += Δx_set[n_rep]
    end;
   (_, idx_detecable_i_a_set) = get_index_detectable(q, L, freq_th, Δx)
else
    global Δx
    @printf("Start processing to get the changes in frequencies.\n")
    @time (Δx_set, x_set) = get_frequency_change(csv_raw, q, L); 
    @printf("Done.\n")
    # Get integrated covariance. This process could be longer (~37min - 7h)
    @printf("Start processing to get the integrated covariancies.\n")
    @time (icov_set, Δx, idx_detecable_i_a_set, idx_detectable) = get_integrated_cov(q, L, Δx_set, x_set, csv_raw, freq_th);
    @printf("Done.\n")
    @printf("Start processing to get the correction in the integrated covariance with linear interpolation.\n")
    @time iΔxΔxT_set = get_correction_with_linear_interpolation(Δx, x_set, csv_raw, freq_th);
    @printf("Done.\n")
    if(flag_out)
        @printf("Start outputting the integrated covariance and the correction in the integrated covariance.\n")
        output_icov_iΔxΔxT(icov_set, iΔxΔxT_set, Δx_set, dir_out)
        @printf("Done.\n")
    
        @printf("Start outputting the changes in frequencies.\n")
        output_frequency_change(Δx_set, x_set, idx_detectable, idx_detecable_i_a_set, dir_out)
        @printf("Done.\n")
    end
end

# Run inference; get selection and epistasis . 
Leff_1st = length(idx_detecable_i_a_set)
idx_reduced = abs.(Δx) .> freq_th
icov = zeros(Leff_1st, Leff_1st)
iΔxΔxT = zeros(Leff_1st, Leff_1st)
Δx_reduced = Δx[idx_reduced] # this contains multiple replications.
for n_rep in 1:length(ids_replicate)
    global icov, iΔxΔxT
    # Get the grouped integrated covariance and the correction in the integrated covariance across replications.
    icov += icov_set[n_rep]
    iΔxΔxT += iΔxΔxT_set[n_rep]
end

for γ1 in γ_set
    s_set = []
    global icov, iΔxΔxT, Δx_reduced
    γI = diagm( 0=>γ1*ones(Leff_1st) )
    for n_rep in 1:length(ids_replicate)
        @printf("Start inference with rep. %d, reg=%.2e .\n", n_rep, γ1)
        @time s_MPL = (icov_set[n_rep] + (1.0/6) * iΔxΔxT_set[n_rep] + γI) \ Δx_set[n_rep][idx_reduced]; 
        @printf("Done.\n")
        push!(s_set, copy(s_MPL))
    end
    @printf("Start inference for grouped statistics.\n")
    @time s_MPL_grouped = (icov +(1.0/6) * iΔxΔxT + γI) \ Δx_reduced;
    @printf("Done.\n")
    # =================================================== : Process End. #
    ##############  Output the results ##############
    # Output the detecable sites and states
    type_set, i_set, a_set, v_set = [], [], [], []
    WT_indicator_i  = []
    ref_seq_cat = [amino_acid_to_number[c] for c in ref_seq];

    for id_ia in 1:length(idx_detecable_i_a_set)
        ia = idx_detecable_i_a_set[id_ia]
        i,a = ia
        push!(type_set, "selection")
        push!(i_set, @sprintf("%d", i)); 
        push!(a_set, @sprintf("%s", number_to_amino_acid[a]));
        push!(WT_indicator_i, ref_seq_cat[i] == a); 
    end;
    
    df = DataFrame( 
        types=type_set,
        position_i=i_set, 
        AA_i=a_set, 
        WT_indicator_i=WT_indicator_i, 
        inference_grouped=[@sprintf("%.6e", x) for x in s_MPL_grouped],
        );
    for i_rep in 1:length(ids_replicate)
        df[!, @sprintf("inference_rep%d", i_rep)] = [@sprintf("%.6e", x) for x in s_set[i_rep]] 
    end
        CSV.write(@sprintf("%sdetecable_sites_states_inferred_parameters_gamma1-%.2e.csv", dir_out, γ1), df); 
end
