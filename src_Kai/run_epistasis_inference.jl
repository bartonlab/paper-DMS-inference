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

include("./tools_for_epistasis_inference.jl")

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
fname_time_steps = ARGS[11] #
# ==============================================#
L, q = length(split(ref_seq, "")), 21
LLhalf = Int(L * (L - 1)/2); 
qL = q*L; qq = q^2
x_rank = qL + qq * LLhalf

@time csv_raw = DataFrame(CSV.File(dir_in * file_csv_in));
(ids_replicate, ids_rounds) = get_replication_round_ids(names(csv_raw));
γ_set = get_reg_set(ids_replicate, ids_rounds, csv_raw)

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
    (_, idx_detecable_i_a_set, idx_detecable_ij_ab_set) = get_index_detectable(q, L, freq_th, Δx)
else
    global Δx
    @printf("Start processing to get the changes in frequencies.\n")
    @time (Δx_set, x_set) = get_frequency_change(csv_raw, q, L); 
    @printf("Done.\n")
    # Get integrated covariance. This process could be longer (~37min - 7h)
    @printf("Start processing to get the integrated covariancies.\n")
    @time (icov_set, Δx, idx_detecable_i_a_set, idx_detecable_ij_ab_set) = get_integrated_cov(q, L, Δx_set, x_set, csv_raw, fname_time_steps, freq_th);
    @printf("Done.\n")
    @printf("Start processing to get the correction in the integrated covariance with linear interpolation.\n")
    @time iΔxΔxT_set = get_correction_with_linear_interpolation(Δx, x_set, csv_raw, freq_th);
    @printf("Done.\n")
    if(flag_out)
        @printf("Start outputting the integrated covariance and the correction in the integrated covariance.\n")
        output_icov_iΔxΔxT(icov_set, iΔxΔxT_set, Δx_set, dir_out)
        @printf("Done.\n")
    end
end

# Run inference; get selection and epistasis . 
Leff_1st, Leff_2nd = length(idx_detecable_i_a_set), length(idx_detecable_ij_ab_set)
idx_reduced = abs.(Δx) .> freq_th
icov = zeros(Leff_1st + Leff_2nd, Leff_1st + Leff_2nd)
iΔxΔxT = zeros(Leff_1st + Leff_2nd, Leff_1st + Leff_2nd)
Δx_reduced = Δx[idx_reduced] # this contains multiple replications.
for n_rep in 1:length(ids_replicate)
    global icov, iΔxΔxT
    # Get the grouped integrated covariance and the correction in the integrated covariance across replications.
    icov += icov_set[n_rep]
    iΔxΔxT += iΔxΔxT_set[n_rep]
end

for (γ1, γ2) in zip(γ_set, γ_set)
    s_set = []
    global icov, iΔxΔxT, Δx_reduced
    γI = diagm(0=>[γ1*ones(Leff_1st); γ2*ones(Leff_2nd)])
    for n_rep in 1:length(ids_replicate)
        @printf("Start inference with rep. %d.\n", n_rep)
        @time s_MPL = (icov_set[n_rep] + (1.0/6) * iΔxΔxT_set[n_rep] + γI) \ Δx_set[n_rep][idx_reduced]; 
        @printf("Done.\n")
        push!(s_set, copy(s_MPL))
    end
    @printf("Start inference for grouped statistics.\n")
    @time s_MPL_grouped = (icov +(1.0/6) * iΔxΔxT + γI) \ Δx_reduced;
    @printf("Done.\n")
    # Free memory:
    #icov=[]; iΔxΔxT=[]; Δx_reduced=[]; Δx=[]; x_set=[]; Δx_set=[];
    # =================================================== : Process End. #
    ##############  Output the results ##############
    # Output the detecable sites and states
    type_set, i_set, j_set, a_set, b_set, v_set = [], [], [], [], [], []
    WT_indicator_i, WT_indicator_j = [], []
    ref_seq_cat = [amino_acid_to_number[c] for c in ref_seq];

    for id_ia in 1:length(idx_detecable_i_a_set)
        ia = idx_detecable_i_a_set[id_ia]
        i,a = ia
        push!(type_set, "selection")
        push!(i_set, @sprintf("%d", i)); push!(j_set, "NaN")
        push!(a_set, @sprintf("%s", number_to_amino_acid[a])); push!(b_set, "NaN")
        push!(WT_indicator_i, ref_seq_cat[i] == a); push!(WT_indicator_j, "NaN")
    end;
    for id_ijab in 1:length(idx_detecable_ij_ab_set)
        ijab = idx_detecable_ij_ab_set[id_ijab]
        (ij,ab) = ijab
        i,j = ij; a,b = ab
        push!(type_set, "epistasis")
        push!(i_set, @sprintf("%d", i)); push!(j_set, @sprintf("%d", j))
        push!(a_set,  @sprintf("%s", number_to_amino_acid[a])); push!(b_set, @sprintf("%s", number_to_amino_acid[b]))
        push!(WT_indicator_i, ref_seq_cat[i] == a); push!(WT_indicator_j, ref_seq_cat[j] == b)
    end;
    df = DataFrame( 
        types=type_set,
        position_i=i_set, position_j=j_set,
        AA_i=a_set, AA_j=b_set,
        WT_indicator_i=WT_indicator_i, WT_indicator_j=WT_indicator_j,
        inference_grouped=[@sprintf("%.6e", x) for x in s_MPL_grouped],
        inference_rep1=[@sprintf("%.6e", x) for x in s_set[1]],
        inference_rep2=[@sprintf("%.6e", x) for x in s_set[2]]
        );
    CSV.write(@sprintf("%sdetecable_sites_states_inferred_parameters_gamma1-%.2e_gamma2-%.2e.csv", dir_out, γ1, γ2), df); 
end
