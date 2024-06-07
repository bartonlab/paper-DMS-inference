stepfunc(x) = x>0 ? 1 : 0
G(i,j,L) = stepfunc(j-i) * Int((i-1)*(L-i/2.0) + j-i);
kr(a, b) = Int(a==b)
km(i,a,q) = (i-1)*q+a;
aa_dict = Dict(
    "Ala" => "A", "Arg" => "R", "Asn" => "N", "Asp" => "D",
    "Cys" => "C", "Glu" => "E", "Gln" => "Q", "Gly" => "G",
    "His" => "H", "Ile" => "I", "Leu" => "L", "Lys" => "K",
    "Met" => "M", "Phe" => "F", "Pro" => "P", "Ser" => "S",
    "Thr" => "T", "Trp" => "W", "Tyr" => "Y", "Val" => "V",
    "Ter" => "*" # Ter is a termination codon (stop codon)
);
# A dictionary to map amino acid letters to numbers
amino_acid_to_number = Dict(
    'A' => 1, 'C' => 2, 'D' => 3, 'E' => 4,
    'F' => 5, 'G' => 6, 'H' => 7, 'I' => 8,
    'K' => 9, 'L' => 10, 'M' => 11, 'N' => 12,
    'P' => 13, 'Q' => 14, 'R' => 15, 'S' => 16,
    'T' => 17, 'V' => 18, 'W' => 19, 'Y' => 20,
    '*' => 21 # '*' represents a stop codon
);

# A reverse dictionary that maps a number to an amino-acid letter.
number_to_amino_acid = Dict(value => key for (key, value) in amino_acid_to_number)

function get_replication_round_ids(list)
    # Initialize sets for storing unique ids
    before_c_set = Set{Int}()
    after_c_set = Set{Int}()
    # Iterate over the list and extract ids using regular expressions
    for item in list
        this_match = match(r"(\d+)_c_(\d+)", item)
        if this_match !== nothing
            push!(before_c_set, parse(Int, this_match.captures[1]))
            push!(after_c_set, parse(Int, this_match.captures[2]))
        end
    end
    before_c_set = sort([Int(x) for x in before_c_set])
    after_c_set = sort([Int(x) for x in after_c_set])
    return (before_c_set, after_c_set)
end;

function reconstruct_list(before_list, after_list)
    new_list = String[]
    for before in before_list
        for after in after_list
            push!(new_list, string(before, "_c_", after))
        end
    end
    return new_list
end

function tuple_index_set(q, L)
    index_i_a_set = []
    for i in 1:L for a in 1:q
        push!(index_i_a_set, [(i), (a)])
    end end
    index_ij_ab_set = []
    for i in 1:L 
        for j in (i+1):L 
            for a in 1:q
                for b in 1:q
                    push!(index_ij_ab_set, [(i,j), (a,b)])
                end
            end
        end
    end
    return (index_i_a_set, index_ij_ab_set)
end;

function get_frequency_change(csv_raw, q, L)
    n_seq_max = length(csv_raw.sequence)
    LLhalf = Int(L * (L - 1)/2); 
    qL = q*L; qq = q^2
    x_rank = qL + qq * LLhalf
    Δx_set = []; x_set = []
    (ids_replicate, ids_rounds) = get_replication_round_ids(names(csv_raw))
    for n_rep in 1:length(ids_replicate)
        # Get headers for a specific replication.
        rep_round_headder = reconstruct_list(ids_replicate[n_rep], ids_rounds);
        n_round = length(rep_round_headder)
        n_seq_tot_set = [sum(csv_raw[:, Symbol(x)]) for x in rep_round_headder]
        scale_set = 1.0 ./ n_seq_tot_set
        # Set variables for the initial and final rounds.
        
        # Start the process to estimate the frequencies.
        Δx = zeros(x_rank); x_each_rounds = zeros(n_round, x_rank)
        for id_seq in 1:n_seq_max
            seq = csv_raw.sequence[id_seq]
            seq_cat = [amino_acid_to_number[c] for c in seq] # having one out of 1 - 21. 
            # Get the counts for the final and initial rounds.
            num_seq_set = [csv_raw[id_seq, Symbol(x)] for x in rep_round_headder]
            # TODO: If we assign WT -> 0, then the number of operations decreases and the matrix becomes sparse.
            coeff = (num_seq_set[end] * scale_set[end] - num_seq_set[1] * scale_set[1])
            for i in 1:L
                a = seq_cat[i]
                Δx[km(i,a,q)] += coeff
                for i_rnd in 1:n_round
                    x_each_rounds[i_rnd, km(i,a,q)] += num_seq_set[i_rnd] * scale_set[i_rnd]
                end
                
                for j in (i+1):L
                    b = seq_cat[j]
                    ξ = qq * (G(i,j,L) - 1) + qL
                    ξ += km(a,b,q)
                    Δx[ξ] += coeff
                    for i_rnd in 1:n_round
                        x_each_rounds[i_rnd, ξ] += num_seq_set[i_rnd] * scale_set[i_rnd]
                    end
                end
            end
        end
        push!(Δx_set, copy(Δx)); push!(x_set, copy(x_each_rounds))
    end
    return (Δx_set, x_set)
end;

function get_index_detectable(q, L, freq_th, Δx)
    qL = q * L
    (index_i_a_set, index_ij_ab_set) = tuple_index_set(q, L)
    idx_detectable = abs.(Δx) .> freq_th
    idx_detecable_i_a_set = index_i_a_set[idx_detectable[1:qL]];
    idx_detecable_ij_ab_set = index_ij_ab_set[idx_detectable[(1+qL):end]];
    return (idx_detectable, idx_detecable_i_a_set, idx_detecable_ij_ab_set)
end

function get_integrated_cov(q, L, Δx_set, x_set, csv_raw, fname_time_steps, freq_th=1e-10)
    qL = q * L
    (ids_replicate, ids_rounds) = get_replication_round_ids(names(csv_raw))
    n_seq_max = length(csv_raw.sequence)

    Δx = zeros(x_rank)
    for n_rep in 1:length(ids_replicate)  
        Δx += Δx_set[n_rep]
    end;
    
    # Suppose all file has the time points in a plain text file.
    t_set = readdlm(fname_time_steps)[:,1]
    Δt_set = zeros(length(t_set))
    Δt_set[1] = 0.5 * (t_set[2] - t_set[1])
    Δt_set[end] = 0.5 * (t_set[end] - t_set[end-1])
    for i in 2:(length(t_set)-1)
        Δt_set[i] = 0.5 * (t_set[i+1] - t_set[i-1]) 
    end
     
    # Compute the integrated covariance matrix. (assume simple piece-wise constant case)
    # TODO: consider piece-wise linear interpolation's case. 
    # Consider only the detectable sites (i,a) where i and a are locus and allele.  
    # To this end we set the effective matrix size as Leff = count(abs.(Δx) .> freq_th);
    (idx_detectable, idx_detecable_i_a_set, idx_detecable_ij_ab_set) = get_index_detectable(q, L, freq_th, Δx) 
    len_idx_detecable_i_a_set = length(idx_detecable_i_a_set)
    len_idx_detecable_ij_ab_set = length(idx_detecable_ij_ab_set)
    
    Leff = count( abs.(Δx) .> freq_th)
    L_1st_eff = length(idx_detecable_i_a_set)
    L_2nd_eff = length(idx_detecable_ij_ab_set)
    @assert Leff == (L_1st_eff + L_2nd_eff)
    @printf("Leff = %d (1st:%d, 2nd:%d)\n", Leff, L_1st_eff, L_2nd_eff)
    
    icov_set = [];
    for n_rep in 1:length(ids_replicate)
        push!(icov_set, zeros(Leff, Leff) )
    end;
        
    for n_rep in 1:length(ids_replicate)        
        # Get headers for a specific replication.
        rep_round_headder = reconstruct_list(ids_replicate[n_rep], ids_rounds);
        n_round = length(rep_round_headder)
        n_seq_tot_set = [sum(csv_raw[:, Symbol(x)]) for x in rep_round_headder]
        scale_set = 1.0 ./ n_seq_tot_set
        
        x_detecable_i_a_set = zeros(n_round, L_1st_eff)
        x_detecable_ij_ab_set = zeros(n_round, L_2nd_eff)
        for i_rnd in 1:n_round
            x_detecable_i_a_set[i_rnd, :] = copy(x_set[n_rep][i_rnd, 1:qL][idx_detectable[1:qL]])
            x_detecable_ij_ab_set[i_rnd, :] = copy(x_set[n_rep][i_rnd, (1+qL):end][idx_detectable[(1+qL):end]])
        end
        
        # Note, in the following processes, we assume Δt = 1 for all time points. 
        for id_seq in 1:n_seq_max            
            seq = csv_raw.sequence[id_seq]
            seq_cat = [amino_acid_to_number[c] for c in seq] # having one out of 1 - 21. 
            num_seq_set = [csv_raw[id_seq, Symbol(x)] for x in rep_round_headder]
            coeff_set = num_seq_set .* scale_set
            # It isn't necessary to sum the outer products each time; this operation can be absolved to the summed coefficient.
            
            coeff_set_sum = sum(coeff_set .* Δt_set)
            g_ia = zeros(L_1st_eff) 
            for id_ia in 1:len_idx_detecable_i_a_set
                (i,a) = idx_detecable_i_a_set[id_ia]
                g_ia[id_ia] = kr(a, seq_cat[i])
            end
            g_ia *= coeff_set_sum
            
            g_klcd = zeros(L_2nd_eff) 
            for id_klcd in 1:len_idx_detecable_ij_ab_set
                klcd = idx_detecable_ij_ab_set[id_klcd]
                (kl,cd) = klcd
                k,l = kl; c,d = cd
                g_kc = kr(c, seq_cat[k])
                g_ld = kr(d, seq_cat[l])
                g_klcd[id_klcd] = g_kc * g_ld
            end  
            g_klcd *= coeff_set_sum
            
            # 2nd order cov.:
            for s in 1:L_1st_eff
                # Sum only non-zero elements
                if(g_ia[s] != 0) # Note g_ia is either 0 or 1.
                    icov_set[n_rep][s, 1:L_1st_eff] += g_ia # Julia is column-major!
                end
                
            end
            # 3rd order cov.:
            for s in 1:L_2nd_eff
                # Sum only non-zero elements
                if(g_klcd[s] != 0)
                    icov_set[n_rep][s+L_1st_eff, 1:L_1st_eff] += g_ia # Julia is column-major!
                end
            end
            # 4th order cov.:
            for s in 1:L_2nd_eff
                # Sum only non-zero elements
                if(g_klcd[s] != 0)
                    icov_set[n_rep][s+L_1st_eff, (1+L_1st_eff):end] += g_klcd
                end
            end 
        end        
        icov_set[n_rep][1:L_1st_eff, (1+L_1st_eff):end] = copy(icov_set[n_rep][(1+L_1st_eff):end, 1:L_1st_eff]')
        @printf("Rep.%d, Symmetry check: %s\n", n_rep, issymmetric(icov_set[n_rep])) 
        
        # Subtracting the moments
        for i_rnd in 1:n_round
            w = Δt_set[i_rnd]
            x_ia = x_detecable_i_a_set[i_rnd, :]
            x_klcd = x_detecable_ij_ab_set[i_rnd, :]
            # 2nd order cov.:
            icov_set[n_rep][1:L_1st_eff, 1:L_1st_eff] -= (w * x_ia) * x_ia'
            # 3rd order cov.:
            icov_set[n_rep][1:L_1st_eff, (1+L_1st_eff):end] -= (w * x_ia) * x_klcd'
            icov_set[n_rep][(1+L_1st_eff):end, 1:L_1st_eff] -= x_klcd * (w * x_ia)'
            # 4th order cov.:
            icov_set[n_rep][(1+L_1st_eff):end, (1+L_1st_eff):end] -= (w * x_klcd) * x_klcd'
        end
    end
    return (icov_set, Δx, idx_detecable_i_a_set, idx_detecable_ij_ab_set)
end;

# Let C^l and C^c are integrated covariance with piecewise linear and piecewise constant interpolation at a temporal time (between t and t+1)
# Then we get C^l = C^c + 1/6 * ΔxΔx^\top, where Δx = x(t+1) - x(t). 
# Note, this expression is valid for both off-diagonal and diagonal elements. 
# When we consider a half of covariance at t=0 and t=t_K, then integrated covariance with linear interpolation iC^l is:
# iC^l = iC^c + 1/6∑_{k=0}^{K+1} Δx(k,k+1)Δx(k,k+1)^\top)
function get_correction_with_linear_interpolation(Δx, x_set, csv_raw, freq_th=1e-10)
    (ids_replicate, ids_rounds) = get_replication_round_ids(names(csv_raw))
    
    idx_reduced = abs.(Δx) .> freq_th
    Leff = count( idx_reduced )
    iΔxΔxT_set = []
    for n_rep in 1:length(ids_replicate)
        iΔxΔxT = zeros(Leff, Leff)
        rep_round_headder = reconstruct_list(ids_replicate[n_rep], ids_rounds);
        n_round = length(rep_round_headder)
        for i_rnd in 2:n_round
            Δx_temp = x_set[n_rep][i_rnd, idx_reduced] - x_set[n_rep][i_rnd-1, idx_reduced]
            iΔxΔxT += Δx_temp * Δx_temp'
        end
        push!(iΔxΔxT_set, copy(iΔxΔxT))
    end
    return iΔxΔxT_set
end;

function output_icov_iΔxΔxT(icov_set, iΔxΔxT_set, Δx_set, dir_out)
    for n_rep in 1:length(icov_set)
        fout=open(@sprintf("%sicov_rep-%s.txt", dir_out, n_rep), "w")
        Leff = size(icov_set[n_rep],1)
        for i in 1:Leff
            out_str = join([@sprintf("%.6e", x) for x in icov_set[n_rep][i, :]], " ")
            println(fout, out_str)
        end
        close(fout)
    end
    for n_rep in 1:length(iΔxΔxT_set)
        fout=open(@sprintf("%siDxDT_rep-%s.txt", dir_out, n_rep), "w")
        Leff = size(iΔxΔxT_set[n_rep],1)
        for i in 1:Leff
            out_str = join([@sprintf("%.6e", x) for x in iΔxΔxT_set[n_rep][i, :]], " ")
            println(fout, out_str)
        end
        close(fout)
    end
    for n_rep in 1:length(Δx_set)
        fout=open(@sprintf("%sDx_rep-%s.txt", dir_out, n_rep), "w")
        Leff = size(Δx_set[n_rep],1)
        for i in 1:Leff
            out_str = @sprintf("%.6e", Δx_set[n_rep][i])
            println(fout, out_str)
        end
        close(fout)
    end
end

function load_icov_iΔxΔxT(dir_load, ids_replicate)
    icov_set, iΔxΔxT_set, Δx_set = [], [], []
    for n_rep in 1:length(ids_replicate)
        icov = readdlm(@sprintf("%sicov_rep-%d.txt", dir_load, n_rep))
        iΔxΔxT = readdlm(@sprintf("%siDxDT_rep-%d.txt", dir_load, n_rep))
        Δx = readdlm(@sprintf("%sDx_rep-%d.txt", dir_load, n_rep))
        push!(icov_set, copy(icov))
        push!(iΔxΔxT_set, copy(iΔxΔxT))
        push!(Δx_set, copy(Δx))
    end
    return (icov_set, iΔxΔxT_set, Δx_set) 
end

function get_best_regularization(corrs, gamma_values; corr_cutoff_pct=0.05)
    corr_thresh = (maximum(corrs)^2 - corrs[1]^2) * corr_cutoff_pct
    gamma_opt = 0.1
    if abs(maximum(corrs)^2 - corrs[1]^2) < 0.01
        gamma_opt = gamma_values[1]
    else
        gamma_set = false
        for i in argmax(corrs):-1:2
            if abs((corrs[i]^2 - corrs[i-1]^2) / (log10(gamma_values[i]) - log10(gamma_values[i-1]))) >= corr_thresh
                gamma_opt = gamma_values[i]
                gamma_set = true
                break
            end
        end
        if !gamma_set
            gamma_opt = gamma_values[argmax(corrs)]
        end
    end
    return gamma_opt
end;

function get_gamma_list_from_file_list(fkey_file, fkey_dir, flag_epistasis=false)
    files = Glob.glob(fkey_file * "*.csv", fkey_dir)

    # Extract the <gamma> values from the filenames
    gamma_values = []

    for file in files
        match_this = match(r"parameters_gamma1-(.*)\.csv", file)
        if(flag_epistasis)
            match_this = match(r"parameters_gamma1-(.*)_gamma2-(.*)\.csv", file)
        end
        if match_this !== nothing
            gamma_value = match_this.captures[1]
            push!(gamma_values, gamma_value)
        end
    end
    
    idx = sortperm( parse.(Float64, gamma_values))
    gamma_values = copy(gamma_values[idx])
    return gamma_values
end;

function get_Pearson_Spearman_gamma(gamma_values, fkey_file, fkey_dir)
    Pearson_set = []; Spearman_set = [];
    vec_set_K1, vec_set_K2 = [], []
    for x in gamma_values
        file_s_e = fkey_dir * fkey_file * x * ".csv"
        csv_selc = DataFrame(CSV.File(file_s_e));
        s1_temp = csv_selc.inference_rep1
        s2_temp = csv_selc.inference_rep2    
        cor_P = cor(s1_temp, s2_temp)
        cor_S = corspearman(float.(s1_temp), float.(s2_temp))
        #@printf("%s %.2e, %.2e\n", x, cor_P, cor_S)
        push!(Pearson_set, cor_P)
        push!(Spearman_set, cor_S)
    end;
    return (Pearson_set, Spearman_set)
end;

function get_reg_set(ids_replicate, ids_rounds, csv_raw, end_exp = 4, num_values = 20)
    rep_round_headder = reconstruct_list(ids_replicate, ids_rounds);
    reads_set = [sum(csv_raw[:, Symbol(x)]) for x in rep_round_headder]
    reads_max = maximum(reads_set);
    start_exp = log10(1/reads_max)
    gamma_values = 10 .^ range(start_exp, stop=end_exp, length=num_values)
    return gamma_values
end;