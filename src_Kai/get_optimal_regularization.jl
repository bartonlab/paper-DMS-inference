using Statistics
using StatsBase
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
# ==============================================#
L, q = length(split(ref_seq, "")), 21
LLhalf = Int(L * (L - 1)/2); 
qL = q*L; qq = q^2
x_rank = qL + qq * LLhalf
γ_sweep_set = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-6]  # This is temporal treatment to reproduce the past results. 
#  Process Start : ============================================== #
@time csv_raw = DataFrame(CSV.File(dir_in * file_csv_in));
(ids_replicate, ids_rounds) = get_replication_round_ids(names(csv_raw));

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
    @time (icov_set, Δx, idx_detecable_i_a_set, idx_detecable_ij_ab_set) = get_integrated_cov(q, L, Δx_set, x_set, csv_raw, freq_th);
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
Pearson_set = []; Spearman_set = [] 
for γ_temp in γ_sweep_set 
    γI = diagm(0=>[γ_temp*ones(Leff_1st); γ_temp*ones(Leff_2nd)])
    n_rep = 1
    s_MPL_1 = (icov_set[n_rep] + (1.0/6) * iΔxΔxT_set[n_rep] + γI) \ Δx_set[n_rep][idx_reduced];  
    n_rep = 2
    s_MPL_2 = (icov_set[n_rep] + (1.0/6) * iΔxΔxT_set[n_rep] + γI) \ Δx_set[n_rep][idx_reduced];  
    fout = open(dir_out * "epistasis_gamma-$(γ_temp).txt", "w")
    for (x,y) in zip(s_MPL_1, s_MPL_2)
        println(fout, x, " ", y)
    end
    close(fout)
    push!(Pearson_set, cor(s_MPL_1, s_MPL_2))
    push!(Spearman_set, corspearman(s_MPL_1, s_MPL_2))
end

γ_opt = get_best_regularization(Pearson_set, γ_sweep_set)

fout = open(dir_out * "epistasis_gamma_optimal.txt", "w")
for (x,y,z) in zip(γ_sweep_set, Pearson_set, Spearman_set)
    if(x==γ_opt)
        println(fout, x, " ", y, " ", z, " ", "Optimal")
    else
        println(fout, x, " ", y, " ", z, "_")
    end
end
close(fout)
# ============================================== : Process End #