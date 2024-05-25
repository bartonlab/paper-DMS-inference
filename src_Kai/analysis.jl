km(i,a,q) = (i-1) * q + a;

CODON2AA = Dict(
    "ATA" => "I", "ATC" => "I", "ATT" => "I", "ATG" => "M",
    "ACA" => "T", "ACC" => "T", "ACG" => "T", "ACT" => "T",
    "AAC" => "N", "AAT" => "N", "AAA" => "K", "AAG" => "K",
    "AGC" => "S", "AGT" => "S", "AGA" => "R", "AGG" => "R",
    "CTA" => "L", "CTC" => "L", "CTG" => "L", "CTT" => "L",
    "CCA" => "P", "CCC" => "P", "CCG" => "P", "CCT" => "P",
    "CAC" => "H", "CAT" => "H", "CAA" => "Q", "CAG" => "Q",
    "CGA" => "R", "CGC" => "R", "CGG" => "R", "CGT" => "R",
    "GTA" => "V", "GTC" => "V", "GTG" => "V", "GTT" => "V",
    "GCA" => "A", "GCC" => "A", "GCG" => "A", "GCT" => "A",
    "GAC" => "D", "GAT" => "D", "GAA" => "E", "GAG" => "E",
    "GGA" => "G", "GGC" => "G", "GGG" => "G", "GGT" => "G",
    "TCA" => "S", "TCC" => "S", "TCG" => "S", "TCT" => "S",
    "TTC" => "F", "TTT" => "F", "TTA" => "L", "TTG" => "L",
    "TAC" => "Y", "TAT" => "Y", "TAA" => "*", "TAG" => "*",
    "TGC" => "C", "TGT" => "C", "TGA" => "*", "TGG" => "W"
);
AA_set = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "*", "-"];
aa_to_num = Dict{String, Int}()
for (index, aa) in enumerate(AA_set)
    aa_to_num[aa] = index
end

AA_colors = Dict(
    "A" => "red",
    "C" => "green",
    "D" => "blue",
    "E" => "yellow",
    "F" => "orange",
    "G" => "purple",
    "H" => "cyan",
    "I" => "magenta",
    "K" => "lime",
    "L" => "pink",
    "M" => "teal",
    "N" => "brown",
    "P" => "navy",
    "Q" => "olive",
    "R" => "maroon",
    "S" => "grey",
    "T" => "violet",
    "V" => "gold",
    "W" => "silver",
    "Y" => "black",
    "*" => "white"
);

amino_acid_colors_DMS = Dict(
    "A" => "pink", "V" => "green", "I" => "green", "L" => "green", "M" => "green", 
    "F" => "brown", "Y" => "brown", "W" => "brown",
    "S" => "orange", "T" => "orange", "N" => "purple", "Q" => "purple",
    "K" => "blue", "R" => "blue", "H" => "blue",
    "D" => "red", "E" => "red",
    "C" => "orange", "G" => "pink", "P" => "green",
    "*" => "White" # Stop codon
);

amino_acid_markers = Dict(
    "A" => "diamond",  # pink
    "V" => "circle",   # green
    "I" => "square",   # green
    "L" => "utriangle", # green
    "M" => "star",     # green
    "F" => "circle",   # brown
    "Y" => "square",   # brown
    "W" => "utriangle", # brown
    "S" => "circle",   # orange
    "T" => "square",   # orange
    "N" => "circle",   # purple
    "Q" => "square",   # purple
    "K" => "circle",   # blue
    "R" => "square",   # blue
    "H" => "utriangle", # blue
    "D" => "circle",   # red
    "E" => "square",   # red
    "C" => "utriangle", # orange
    "G" => "circle",    # pink
    "P" => "hexagon",  # green
    "*" => "star"      # White (Stop codon)
);

#Since the for mat of these files can be diferent as never observed mutation could appear in another process, 
function return_corresponding_selections(inference_raw1, inference_raw2, 
    type_col1, type_col2, pos1_i, pos2_i, AA1_i, AA2_i)
s1, s2 = [], []
pos_set_temp, aa_set_temp = [], []
for i in 1:L_eff
    idx1 = inference_raw1[:, Symbol(pos1_i)] .== i
    idx2 = inference_raw2[:, Symbol(pos2_i)] .== i
    csv_compact1 = inference_raw1[idx1, :]
    csv_compact2 = inference_raw2[idx2, :]
    aa_set1 = [string(x) for x in csv_compact1[:, Symbol(AA1_i)]]
    aa_set2 = [string(x) for x in csv_compact2[:, Symbol(AA2_i)]]
    common_aa = intersect( aa_set1, aa_set2 )
    for a in common_aa
        idx1_a = aa_set1 .== a
        idx2_a = aa_set2 .== a
        push!(s1, csv_compact1[idx1_a, Symbol(type_col1)][1])
        push!(s2, csv_compact2[idx2_a, Symbol(type_col2)][1])
        push!(pos_set_temp, i)
        push!(aa_set_temp, a)
    end
end;
return (s1, s2, pos_set_temp, aa_set_temp)
end;

#Since the for mat of these files can be diferent as never observed mutation could appear in another process, 
function return_corresponding_selections_temp(inference_raw1, inference_raw2, 
    type_col1, type_col2, pos1_i, pos2_i, AA1_i, AA2_i)
pos_set_temp, aa_set_temp = [], []
s1, s2 = [], []

idx_set_csv1, idx_set_csv2 = [], [];
n_raw1_max, n_raw2_max = length(inference_raw1[:, 1]), length(inference_raw2[:, 1])
for i in 1:L_eff
    idx1 = inference_raw1[:, Symbol(pos1_i)] .== i
    idx2 = inference_raw2[:, Symbol(pos2_i)] .== i
    actual_idx1, actual_idx2 = collect(1:n_raw1_max)[idx1], collect(1:n_raw2_max)[idx2]
    
    csv_compact1 = inference_raw1[idx1, :]
    csv_compact2 = inference_raw2[idx2, :]
    aa_set1 = [string(x) for x in csv_compact1[:, Symbol(AA1_i)]]
    aa_set2 = [string(x) for x in csv_compact2[:, Symbol(AA2_i)]]
    common_aa = intersect( aa_set1, aa_set2 )
    for a in common_aa
        idx1_a = aa_set1 .== a
        idx2_a = aa_set2 .== a
        
        actual_idx1_aa, actual_idx2_aa = actual_idx1[idx1_a][1], actual_idx2[idx2_a][1]
        push!(idx_set_csv1, actual_idx1_aa)
        push!(idx_set_csv2, actual_idx2_aa)
        push!(s1, csv_compact1[idx1_a, Symbol(type_col1)][1])
        push!(s2, csv_compact2[idx2_a, Symbol(type_col2)][1])
        push!(pos_set_temp, i)
        push!(aa_set_temp, a)
    end
end;
return (s1, s2, pos_set_temp, aa_set_temp, idx_set_csv1, idx_set_csv2)
end;

function rearrange_selection_preference(L_eff, preference_raw_in, inference_raw_in)
    pref_temp_vec, selec_temp_vec = [], []
    for i in 1:L_eff
        idx1 = preference_raw_in.position .== i
        idx2 = inference_raw_in.position_i .== i
        preference_compact_temp = preference_raw_in[idx1, :]
        inference_compact_temp = inference_raw_in[idx2, :] 
        aa_comon = intersect(preference_compact_temp.AA_i, inference_compact_temp.AA_i)
        for a in aa_comon
            idx1_a = preference_compact_temp.AA_i .== a
            idx2_a = inference_compact_temp.AA_i .== a 
            push!(pref_temp_vec, preference_compact_temp.preference_av[idx1_a][1])
            push!(selec_temp_vec, inference_compact_temp.inference_grouped[idx2_a][1])
        end
    end;
    #minimum(pref_temp_vec[pref_temp_vec .> 0])
    pref_temp_vec_filter = copy(pref_temp_vec)
    pref_temp_vec_filter[pref_temp_vec .== 0] .= minimum(pref_temp_vec[pref_temp_vec .> 0]) * 0.9;
    return (pref_temp_vec_filter, selec_temp_vec)
end;
#(pref_vec_filter, selec_vec) = rearrange_selection_preference(L_eff, preference_raw, inference_raw)
#(pref_vec_filter_temp, selec_vec_temp) = rearrange_selection_preference(L_eff, preference_raw_temp, inference_raw_temp);

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
        #aa_num_set_temp = [aa_to_num[x[1]] for x in split(csv_count_raw_in.sequence[n], "")]
        aa_num_set_temp = [aa_to_num[x] for x in split(csv_count_raw_in.sequence[n], "")]
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
    A_denom = tt_mat + γ*I
    inv_A_denom = inv(A_denom)
    
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


function get_SL(n_R, q, L_max, n_T, freq_mat_reformat,γ_SL = 1e-3)
    selection_set = zeros(n_R, q*L_max)
    selection_grouped = zeros(q*L_max)
    for i in 1:L_max
        for i_a in 1:q 
            Δ_x_av, x_var_av = 0.0, 0.0
            for i_r in 1:n_R
                Δx_temp = freq_mat_reformat[i_r, end, km(i, i_a, q)] - freq_mat_reformat[i_r, 1, km(i, i_a, q)]
                Δ_x_av += Δx_temp
                x_var_temp = 0.0
                for k in 1:(n_T-1)
                    x_mean = 0.5 * ( freq_mat_reformat[ i_r, k+1, km(i, i_a, q) ] + freq_mat_reformat[ i_r, k, km(i, i_a, q) ] )
                    x_var_temp += x_mean * ( 1.0 - x_mean )
                end
                x_var_av += x_var_temp
                selection_set[i_r, km(i, i_a, q) ] = Δx_temp / (x_var_temp + γ_SL)
            end
            selection_grouped[ km(i, i_a, q) ] = Δ_x_av / (x_var_av + γ_SL)
        end
    end;
    return (selection_set, selection_grouped)
end;

function resort_site_amino(csv_selection, AA_set, aa_str, i_str)
    idx_set_out = []
    aa_n_set = csv_selection[:, Symbol(aa_str) ]
    i_n_set = csv_selection[:, Symbol(i_str) ]
    idx_raw_csv = collect(1:length(i_n_set) )
    i_set = sort(unique(i_n_set))
    for i in i_set
        for a in AA_set
            idx_temp = (i_n_set .== i) .&& (aa_n_set .== a)            
            if(count(idx_temp)>0)
                i_raw = idx_raw_csv[idx_temp][1]
                push!(idx_set_out, i_raw)
            end
        end
    end;
    return idx_set_out
end;