#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <random>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <omp.h> 
#include <sys/types.h> 
#include <sys/stat.h> 

#include "Tools/c++/config.h"
#include "Tools/c++/utils.h"
#include "Tools/c++/mat.h"
#include "Tools/c++/timer.h"
#include "Tools/c++/svm.h"
#include <mutex>   // Missing
#include <fstream> // Missing
#include <thread>

using namespace std;

class Logger {
	private:
		static std::ofstream log_file;
		static bool initialized;
		static bool verbose;
		static std::string log_path;
		static std::mutex log_mutex;
		static bool console_output_enabled;
		static int verbosity_level;
	
	public:
		static void init(const std::string& path, bool be_verbose = true, bool enable_console = false) {
			std::lock_guard<std::mutex> lock(log_mutex);
			if (!initialized) {
				// Create directory if it doesn't exist
				std::string dir_path = path.substr(0, path.find_last_of('/'));
				mkdir(dir_path.c_str(), S_IRWXU);
				
				log_path = path;
				log_file.open(log_path, std::ios::out);
				verbose = be_verbose;
				console_output_enabled = enable_console; // Default to false
				initialized = true;
				
				// Write header to log
				time_t now = time(0);
				char* time_str = ctime(&now);
				log_file << "=== SLICE LOG STARTED AT " << (time_str ? time_str : "unknown time") << std::endl;
				log_file << "=== Build Date: " << __DATE__ << " " << __TIME__ << std::endl;
				log_file.flush();
			}
		}
		
		template<typename T>
		static void log(const T& msg, bool print_to_console = true) {
			std::lock_guard<std::mutex> lock(log_mutex);
			if (initialized) {
				log_file << msg << std::endl;
				log_file.flush(); // Force flush to disk
			}
			if (verbose && print_to_console) {
				std::cout << msg << std::endl;
				std::cout.flush(); // Force flush to console
			}
		}

		static void threadLog(const std::string& msg, int level = 1) {
			if (level <= verbosity_level) {
				std::string thread_msg = "Thread " + 
										std::to_string(omp_get_thread_num()) + 
										": " + msg;
				log(thread_msg);
			}
		}
		
		static void enableConsoleOutput(bool enable) {
			std::lock_guard<std::mutex> lock(log_mutex);
			console_output_enabled = enable;
		}
		
		static void close() {
			std::lock_guard<std::mutex> lock(log_mutex);
			if (initialized) {
				log_file.close();
				initialized = false;
			}
		}

		static void setVerbosityLevel(int level) {
			verbosity_level = level;
		}
};

class Param
{
public:
	_int num_trn;
	_int num_ft;
	_int num_lbl;
	_int num_threads;
	_bool quiet;
	// HNSW Params
	_int M;
	_int efC;
	_int efS;
	_int num_nbrs;
	_int num_io_threads;
	// Discriminative Model Params
	_float classifier_cost;
	_float classifier_threshold;
	_int classifier_maxiter;
	_int classifier_kind;
	//Prediction param
	_float b_gen;
	
	//updated code start
    // Explainability parameters
    _bool generate_explanations;
    _int num_features_to_explain;	
    // Adversarial robustness parameters
    _bool adversarial_training;
    _bool adversarial_defense;
    _float perturbation_strength;
	bool debug;
	//updated code end 

	Param()
	{
		num_trn = 0;
		num_ft = 0;
		num_lbl = 0;
		num_threads = 1;
		quiet = false;
		M = 100;
		efC = 300;
		efS = 300;
		num_nbrs = 300;
		num_io_threads = 20;
		classifier_cost = 1.0;
		classifier_threshold = 1e-6;
		classifier_maxiter = 20;
		classifier_kind = 0;
		b_gen = 0;
		//updated code start
		generate_explanations = true;
        num_features_to_explain = 5;
        
        adversarial_training = true;
        adversarial_defense = true;
        perturbation_strength = 0.1;
		debug = true;
		//updated code end
	}

	Param(string fname)
	{
		check_valid_filename(fname,true);
		ifstream fin;
		fin.open(fname);
		
		fin>>num_trn;
		fin>>num_ft;
		fin>>num_lbl;
		fin>>num_threads;
		fin>>quiet;
		fin>>M;
		fin>>efC;
		fin>>efS;
		fin>>num_nbrs;
		fin>>num_io_threads;
		fin>>classifier_cost;
		fin>>classifier_threshold;
		fin>>classifier_maxiter;
		fin>>classifier_kind;
		fin>>b_gen;
		fin.close();
	}

	void write(string fname)
	{
		check_valid_filename(fname,false);
		ofstream fout;
		fout.open(fname);

		fout<<num_trn<<"\n";
		fout<<num_ft<<"\n";
		fout<<num_lbl<<"\n";
		fout<<num_threads<<"\n";
		fout<<quiet<<"\n";
		fout<<M<<"\n";
		fout<<efC<<"\n";
		fout<<efS<<"\n";
		fout<<num_nbrs<<"\n";
		fout<<num_io_threads<<"\n";
		fout<<classifier_cost<<"\n";
		fout<<classifier_threshold<<"\n";
		fout<<classifier_maxiter<<"\n";
		fout<<classifier_kind<<"\n";
		fout<<b_gen<<"\n";
		//updated code start
		fout<<generate_explanations<<"\n";
		fout<<num_features_to_explain<<"\n";
		fout<<adversarial_training<<"\n";
		fout<<adversarial_defense<<"\n";
		fout<<perturbation_strength<<"\n";
		//updated code end
		fout.close();
	}
	void print()
	{
		cout<<"Number of training examples="<<num_trn<<"\n";
		cout<<"Number of features="<<num_ft<<"\n";
		cout<<"Number of labels="<<num_lbl<<"\n";
		cout<<"Number of train/test threads="<<num_threads<<"\n";
		cout<<"Quiet="<<quiet<<"\n";
		cout<<"M="<<M<<"\n";
		cout<<"efConstruction="<<efC<<"\n";
		cout<<"efSearch="<<efS<<"\n";
		cout<<"Number of nearest neighbors="<<num_nbrs<<"\n";
		cout<<"Number of threads for I/O="<<num_io_threads<<"\n";
		cout<<"Cost co-efficient for discriminative classifier="<<classifier_cost<<"\n";
		cout<<"Threshold for discriminative classifier="<<classifier_threshold<<"\n";
		cout<<"Maximum number of iterations for the discriminative classifier="<<classifier_maxiter<<"\n";
		cout<<"Separator Type="<<classifier_kind<<"\n";
		cout<<"b_gen="<<b_gen<<"\n";
		printf("Debug=%d\n", debug);
	}
};

DMatF* train_slice(DMatF* trn_ft_mat, SMatF* trn_lbl_mat, string model_dir, Param& params, float& train_time);
SMatF* predict_slice(DMatF* tst_ft_mat, DMatF* w_dis, string model_dir, Param& params, float& test_time); 
