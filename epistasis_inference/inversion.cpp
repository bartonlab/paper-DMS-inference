#include <algorithm>
#include <array>
#include <iterator>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <cstring>
#include <unordered_map>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/Dense>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
//#include </Users/zhenchenhong/Downloads/DMS_20210329/Eigen/Sparse>
//#include </Users/zhenchenhong/Downloads/DMS_20210329/Eigen/SparseLU>
typedef Eigen::Triplet<double> Trip;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::MatrixXd dMat;
template <class OutputIt>
OutputIt safe_tokenizer(const std::string & s, char token, OutputIt out){
    std::string::size_type pos = 0, f;  
    while((f = s.find(token, pos)) != std::string::npos){    
        *out++ = s.substr(pos, f - pos);
        pos = f + 1;
    }
    if(pos < s.size())
        *out++ = s.substr(pos);

    return out;
}

struct Variant_Table{
    int replicate;
    int generation;
    std::vector<int> variant;
    int counts;
    double frequency;
};

typedef std::pair<int, int> pairs;
typedef struct Variant_Table Struct; 

void get_lines_num(FILE* file_head, int *line_num){
    char line[128];

    std::string delimiter = ",";

    std::string token;

    while (fgets(line, 128, file_head)){
        *line_num = *line_num + 1;
    }
}

std::vector<std::array<int, 3>> get_lines_info(FILE* file_head){
    char line[128];

    std::vector<std::array<int, 3>> Info_vec;
    std::string delimiter = ",";
    int i = 0;
    int rep_tmp;
    int gen_tmp;
    int cnt_tmp;
    int vec_list[4];

    std::string substr("total_reads");
    fgets(line, 128, file_head);
    std::string token;

    while (fgets(line, 128, file_head)){

        std::string str(line);
        if(str.find(substr)!=std::string::npos){
            //std::cout << line;
            size_t pos = 0;
            i = 0;
            while ((pos = str.find(delimiter)) != std::string::npos) {
                token = str.substr(0, pos);
                if(i != 2){
                    vec_list[i] = stoi(token);
                }
                //std::cout << token <<std::endl;
                i++;
                str.erase(0, pos + delimiter.length());
            }
            vec_list[i] = stoi(str);
            //std::cout << vec_list[0] <<" "<< vec_list[1] <<" "<<vec_list[3] <<" "<<std::endl; 
            Info_vec.push_back({vec_list[0], vec_list[1], vec_list[3]});
        }
    }

    return Info_vec;  //rep gen counts
}

std::vector<std::string> string_split(std::string s, const char delimiter){
    size_t start = 0;
    size_t end   = s.find_first_of(delimiter);
    std::vector<std::string> output;
    
    while (end <= std::string::npos)
    {
        output.emplace_back(s.substr(start, end - start));
        if (end == std::string::npos)
            break;
        start = end + 1;
        end   = s.find_first_of(delimiter, start);
    }
    
    return output;
}

//['*', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
int get_AA_idx(char AA){
    std::map<char, int> AA_idx = {{'*', 0}, {'A', 1}, {'C', 2}, {'D', 3}, {'E', 4}, {'F', 5}, {'G', 6}, 
                                  {'H', 7}, {'I', 8}, {'K', 9}, {'L', 10}, {'M', 11}, {'N', 12}, {'P', 13}, 
                                  {'Q', 14}, {'R', 15}, {'S', 16}, {'T', 17}, {'V', 18}, {'W', 19}, {'Y', 20}};
    int idx = AA_idx[AA];
    return idx;
}


char codon_trans(std::string codon){
 //-------------------------TXX
    if(codon[0] == 'T'){
        if(codon[1] == 'T'){
            if(codon[2] == 'T' || codon[2] == 'C'){
                return 'F';
            }
            else if(codon[2] == 'A' || codon[2] == 'G'){
                return 'L';
            }
        }
        else if(codon[1] == 'A'){
            if(codon[2] == 'T' || codon[2] == 'C'){
                return 'Y';
            }
            else if(codon[2] == 'A' || codon[2] == 'G'){
                return '*';
            }
        }
        else if(codon[1] == 'C'){
            return 'S';
        }
        else if(codon[1] == 'G'){
            if(codon[2] == 'T' || codon[2] == 'C'){
                return 'C';
            }
            else if(codon[2] == 'G'){
                return 'W';
            }
            else if(codon[2] == 'A'){
                return '*';
            }
        }
    }
 //----------------------------------CXX
    else if(codon[0] == 'C'){
        if(codon[1] == 'T'){
            return 'L';
        }
        else if(codon[1] == 'C'){
            return 'P';
        }
        else if(codon[1] == 'A'){
            if(codon[2] == 'T' || codon[2] == 'C'){
                return 'H';
            }
            else if(codon[2] == 'G' || codon[2] == 'A'){
                return 'Q';
            }
        }
        else if(codon[1] == 'G'){
            return 'R';
        }
    }
 //----------------------------------AXX
    else if(codon[0] == 'A'){
        if(codon[1] == 'T'){
            if (codon[2] == 'G'){
                return 'M';
            }
            else{
                return 'I';
            }
        }
        else if(codon[1] == 'C'){
            return 'T';
        }
        else if(codon[1] == 'A'){
            if(codon[2] == 'T' || codon[2] == 'C'){
                return 'N';
            }
            else if(codon[2] == 'G' || codon[2] == 'A'){
                return 'K';
            }      
        }
        else if(codon[1] == 'G'){
            if(codon[2] == 'T' || codon[2] == 'C'){
                return 'S';
            }
            else if(codon[2] == 'G' || codon[2] == 'A'){
                return 'R';
            }     
        }
    }
 //-----------------------------------------GXX
    else if(codon[0] == 'G'){
        if(codon[1] == 'T'){
            return 'V';
        }
        else if(codon[1] == 'C'){
            return 'A';
        }
        else if(codon[1] == 'A'){
            if(codon[2] == 'T' || codon[2] == 'C'){
                return 'D';
            }
            else if(codon[2] == 'G' || codon[2] == 'A'){
                return 'E';
            }      
        }
        else if(codon[1] == 'G'){
            return 'G';
        }    
    }
    return '*';
}


int mut2idx(char sin_AA, int site_idx){
    int vector_idx;
    int AA_idx = get_AA_idx(sin_AA);
    return (site_idx - 1) * 21 + AA_idx;
}


int get_site(std::string item) {
   std::stringstream variant;
   variant << item; //convert the string s into stringstream
   std::string temp_str;
   int temp_int;
   while(!variant.eof()) {
      variant >> temp_str; //take words into temp_str one by one
      if(std::stringstream(temp_str) >> temp_int) { //try to convert string to int
         //return temp_int;
      }
      //temp_str = ""; //clear temp string
   }
   return temp_int;
}


void printSelectionCoefficients(FILE *output, const std::vector<double> &s) {
    
    for (int i=0;i<s.size();i++) fprintf(output,"%.6e\n",s[i]);
    fflush(output);

}

int main() {
    std::cout << "Start!\n";
    //std::string raw_sequence = "FLIMVD";
    std::string raw_sequence = "DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPR";
    std::ofstream outputFile;
    // create a name for the file output
    std::string filename;
    int seq_length = raw_sequence.length();
    double regularization = 100.0;
    const char* Index_file = "index_matrix.csv";
    const char* Rawdata = "new_count_1.csv";
    const char *raw_seq = raw_sequence.c_str();
    
    int raw_seq_idx[seq_length];
    for(int tmp_idx = 0; tmp_idx < seq_length; tmp_idx++){
        raw_seq_idx[tmp_idx] = tmp_idx * 21 + get_AA_idx(raw_sequence[tmp_idx]);
    }

// read how many lines in data file & Info_vec = (rep,gen,tot_cnt)
    FILE* get_line = fopen(Index_file, "r");
    int matrix_dim = 0;

    get_lines_num(get_line, &matrix_dim);

    std::cout << "Dimension of matrix = " << matrix_dim << std::endl;



    get_line = fopen(Rawdata, "r");
    std::vector<std::array<int, 3>> Info_vec = get_lines_info(get_line);

    int non_elements, matrix_nnz;
    double sparsity;
    std::string file_name;
    const char* multiple_allele_file;
    const char* sindou_allele_file;

    //std::map<std::tuple<int, int>, SpMat> cov_xijkl;
    //std::map<std::tuple<int, int>, dMat> cov_xixj;
    //std::map<std::tuple<int, int>, dMat> freq_vect_full;
    dMat cov_full(matrix_dim, matrix_dim);
    dMat freq_diff(1, matrix_dim);
    std::map< int, std::map<int, double>> matrix_element;
    std::map< int, std::map<int, double>> freq_element;
    double *dx       = new double[matrix_dim];                           // difference between start and end allele frequencies
    double *totalCov = new double[matrix_dim*matrix_dim];                // accumulated allele covariance matrix
    double *freq_vect_full = new double[Info_vec.size()*matrix_dim];
    for (int a=0; a < matrix_dim;a++)            dx[a]       = 0;
    for (int a=0; a < matrix_dim*matrix_dim;a++) totalCov[a] = 0;
    for(int i = 0; i<Info_vec.size(); i++){
    	for (int a=0; a < matrix_dim;a++) freq_vect_full[i*matrix_dim+a] = 0;}

//xi
    //Eigen::MatrixXd x_ = Eigen::MatrixXd::Random(3,3);
//    Eigen::PartialPivLU<Eigen::MatrixXd> lltOfcov = m.partialPivLu(); 
//    Eigen::VectorXd n = Eigen::VectorXd::Random(3);

//    Eigen::VectorXd x= lltOfcov.solve(n);
//    std::cout<<m<<std::endl;
//    std::cout<<n<<std::endl;
//    std::cout<<x<<std::endl;
//    std::cout<<m*x<<std::endl;

    for(int i = 0; i<Info_vec.size(); i++){
        //std::tuple<int, int> rep_gen = std::make_tuple(Info_vec[i][0], Info_vec[i][1]);

        //for (int a=0; a < matrix_dim;a++) freq_vect_full[i*matrix_dim+a] = 0;

        std::cout <<"Allele frequencies import for: Replicate = "<<Info_vec[i][0]<<", Generation = "<<Info_vec[i][1]<< std::endl;

        file_name = "freq_" + std::to_string(Info_vec[i][0])+"_"+std::to_string(Info_vec[i][1])+".csv";
        //file_name = "test_freq.csv";
        sindou_allele_file = file_name.c_str();

        get_line = fopen(sindou_allele_file, "r");

        non_elements = 0;

        get_lines_num(get_line, &non_elements);

        matrix_nnz = non_elements;

        //freq_vect_full[rep_gen].resize(1, matrix_dim);
        //freq_vect_full[rep_gen].reserve(matrix_nnz); 

        char line[128];
        std::string delimiter = ",";
        int row_num, col_num;
        double freq;
        std::string token;
        get_line = fopen(sindou_allele_file, "r");
        //get_line = fopen("test_freq.csv", "r");

        
        int line_num = 0; 
        freq_element.clear();

        while(fgets(line, 128, get_line)){
            line_num++;
            //if(line_num%10000==0){
            //    std::cout << line_num<< std::endl;
            //}
            size_t pos = 0;
            int col_idx = 0;
            std::string s = line;
            while ((pos = s.find(delimiter)) != std::string::npos) {
                token = s.substr(0, pos);
                if(col_idx == 0){
                    row_num = std::stoi(token);
                    //std::cout << token<< "\t";
                }
                s.erase(0, pos + delimiter.length());
                col_idx++;
            }
            freq = std::stod(s);
            freq_vect_full[i*matrix_dim+row_num] += freq;
            //freq_element[0][row_num] = freq;
        }
/*
        for(auto const &ent1 : freq_element) {  // ent1.first is the first key
            for(auto const &ent2 : ent1.second) {  // ent2.first is the second key, ent2.second is the data
                freq_vect_full[rep_gen](ent1.first, ent2.first) = ent2.second;
            }
        } */
        
        if(i<Info_vec.size()-1){//-xi*xj
            //std::cout<<"=="<<std::endl;
            //cov_xixj[rep_gen].resize(matrix_dim,matrix_dim);
            for(int row=0;row<matrix_dim;row++){
	            if(row%1000==0){
	                std::cout << row<< std::endl;
	            }
            	for(int col = 0;col<matrix_dim;col++){
            		totalCov[row*matrix_dim+col] -= freq_vect_full[i*matrix_dim+row] * freq_vect_full[i*matrix_dim+col];
            		//totalCov[col*matrix_dim+row] -= freq_vect_full[i*matrix_dim+row] * freq_vect_full[i*matrix_dim+col];
            		//if(row==26774 && col==502) std::cout << "=="<<totalCov[row*matrix_dim+col]<<" "<<freq_vect_full[i*matrix_dim+row]<<freq_vect_full[i*matrix_dim+col]<< std::endl;
            	}
            }
            //freq_element.clear(); 

            std::cout <<"Covariance matrix import for Replicate = "<<Info_vec[i][0]<<", Generation = "<<Info_vec[i][1]<< std::endl;

            file_name = "multiple_allele_" + std::to_string(Info_vec[i][0])+"_"+std::to_string(Info_vec[i][1])+".csv";

            multiple_allele_file = file_name.c_str();

            get_line = fopen(multiple_allele_file, "r");

            non_elements = 0;

            get_lines_num(get_line, &non_elements);

            matrix_nnz = (non_elements - matrix_dim)*2 + matrix_dim;

            std::cout << "Non-zero elements # = " << matrix_nnz << std::endl;

            sparsity = double(matrix_nnz)*100/double(matrix_dim*matrix_dim);

            std::cout << "Sparsity = " << sparsity <<"%"<< std::endl;

            //cov_xijkl[rep_gen].resize(matrix_dim, matrix_dim);
            //cov_xijkl[rep_gen].reserve(matrix_nnz); 

            char line[128];
            std::string delimiter = ",";
            int row_num, col_num;
            double freq;
            std::string token;
            get_line = fopen(multiple_allele_file, "r");
            //get_line = fopen("test_matrix.csv", "r");

            //matrix_element.clear();
            int line_num = 0; 

            while(fgets(line, 128, get_line)){//xijkl
                line_num++;
                if(line_num%1000000==0){
                    std::cout << line_num<< std::endl;
                }
                size_t pos = 0;
                int col_idx = 0;
                std::string s = line;
                while ((pos = s.find(delimiter)) != std::string::npos) {
                    token = s.substr(0, pos);

                    if(col_idx == 0){
                        row_num = std::stoi(token);
                        //std::cout << token<< "\t";
                    }
                    if(col_idx == 1){
                        col_num = std::stoi(token);
                        //std::cout << token<< "\t";
                    }  

                    s.erase(0, pos + delimiter.length());
                    col_idx++;
                }
                freq = std::stod(s);
                if(row_num == col_num) totalCov[col_num*matrix_dim+row_num] += freq;
                else{
                	totalCov[col_num*matrix_dim+row_num] += freq;
                	totalCov[row_num*matrix_dim+col_num] += freq;
                }
            } 
        }     
    }

    for(int i =0;i<matrix_dim;i++){
    	dx[i] = freq_vect_full[(Info_vec.size()-1)*matrix_dim + i] - freq_vect_full[0*matrix_dim + i];
    	totalCov[i*matrix_dim+i] += regularization;
    }

    // std::cout<<freq_vect_full[0*matrix_dim+26774]<<std::endl;
    // std::cout<<freq_vect_full[0*matrix_dim+502]<<std::endl;
    // std::cout<<freq_vect_full[1*matrix_dim+26774]<<std::endl;
    // std::cout<<freq_vect_full[1*matrix_dim+502]<<std::endl;
    // std::cout<<freq_vect_full[2*matrix_dim+26774]<<std::endl;
    // std::cout<<freq_vect_full[2*matrix_dim+502]<<std::endl;
    // std::cout<<freq_vect_full[3*matrix_dim+26774]<<std::endl;
    // std::cout<<freq_vect_full[3*matrix_dim+502]<<std::endl;
    // std::cout<<totalCov[502*matrix_dim + 26774]<<std::endl;
    // std::cout<<totalCov[26774*matrix_dim + 502]<<std::endl;
    // std::cout<<totalCov[502*matrix_dim + 502]<<std::endl;
    // std::cout<<totalCov[26774*matrix_dim + 26774]<<std::endl;
    // std::cout<<dx[502]<<std::endl;
    // std::cout<<dx[26774]<<std::endl;

    int status;
    std::vector<double> sMAP(matrix_dim,0);

    std::cout<<"prepare vectors"<<std::endl;
    gsl_matrix_view _cov = gsl_matrix_view_array(totalCov, matrix_dim, matrix_dim);   // gsl covariance + Gaussian regularization
    gsl_vector_view  _dx = gsl_vector_view_array(dx, matrix_dim);            // gsl dx vector
    std::cout<<"prepare space"<<std::endl;
    gsl_vector    *_sMAP = gsl_vector_alloc(matrix_dim);                     // maximum a posteriori selection coefficients for each allele
    gsl_permutation  *_p = gsl_permutation_alloc(matrix_dim);
    std::cout<<"decomposition"<<std::endl;
    gsl_linalg_LU_decomp(&_cov.matrix, _p, &status);
    std::cout<<"solve"<<std::endl;
    gsl_linalg_LU_solve(&_cov.matrix, _p, &_dx.vector, _sMAP);
    
    for (int a=0;a<matrix_dim;a++) {
    	sMAP[a] = gsl_vector_get(_sMAP, a);
    }

    FILE *fp;  
    fp = fopen("20220304_rep2_2.txt", "w");
   	std::cout<<"write file"<<std::endl;
    for (int i=0;i<sMAP.size();i++) {
    	fprintf(fp,"%.6e\n",sMAP[i]);
    }

/*
    for(int i =0;i<matrix_dim;i++){
    	dx[i] = freq_vect_full[(Info_vec.size()-1)*matrix_dim + i] - freq_vect_full[0*matrix_dim + i];
    	totalCov[i*matrix_dim+i] += regularization;
    }


    gsl_matrix_view _cov = gsl_matrix_view_array(totalCov, matrix_dim, matrix_dim);   // gsl covariance + Gaussian regularization
    gsl_vector_view  _dx = gsl_vector_view_array(dx, matrix_dim);            // gsl dx vector
    gsl_vector    *_sMAP = gsl_vector_alloc(matrix_dim);                     // maximum a posteriori selection coefficients for each allele
    gsl_permutation  *_p = gsl_permutation_alloc(matrix_dim);
    
    gsl_linalg_LU_decomp(&_cov.matrix, _p, &status);
    gsl_linalg_LU_solve(&_cov.matrix, _p, &_dx.vector, _sMAP);
    
    for (int a=0;a<L;a++) sMAP[a] = gsl_vector_get(_sMAP, a);
*/
   	
/*    
    std::cout<<"finish cumulation"<<std::endl;
    freq_diff += freq_vect_full[std::make_tuple(1, 3)];
    //freq_diff += freq_vect_full[std::make_tuple(1, 1)];
    freq_diff -= freq_vect_full[std::make_tuple(1, 0)];
    std::cout<<"finish difference"<<std::endl;
    //dMat identity(matrix_dim, matrix_dim) = MatrixXd::Identity(4, 3);
    for(int i = 0;i < matrix_dim; i++){
        cov_full(i,i)+=regularization;
    }
    
    freq_vect_full.clear();
    std::ofstream file;
    //std::cout<<cov_full(1233,4964)<<std::endl;
    std::cout<<"record frequency_change"<<std::endl;
 
    file.open("frequency_change_matrix.csv");
    file << freq_diff << std::endl;  
    file.close(); 

    std::cout<<"record covariance_matrix"<<std::endl;
   
    file.open("covariance_matrix.csv");
    for(int i = 0;i < matrix_dim; i++){
        if(i%100==0){std::cout<<i<<std::endl;}
        for(int j=0;j<matrix_dim-1;j++){
            file << cov_full(i,j)<<",";
        }
        file << cov_full(i,matrix_dim-1)<<std::endl;
    }
     
    file.close(); 

    //cov_full.resize(0,0); 


    //freq_diff.resize(0,0);


    std::cout<<"PartialPivLU..."<<std::endl;
    Eigen::PartialPivLU<Eigen::MatrixXd> lltOfcov = cov_full.partialPivLu(); 


    std::cout<<"solving..."<<std::endl;
    Eigen::VectorXd x= lltOfcov.solve(freq_diff);

    std::cout<<x<<std::endl;
    //std::ofstream file;
    file.open("result.csv");
    file << x << std::endl;
    file.close();
*/

    return 0;
}


