#include "slice.h"

using namespace std;
thread_local mt19937 reng; // random number generator used during training 

_int get_rand_num( _int siz )
{
	_llint r = reng();
	_int ans = r % siz;
	return ans;
}


#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)
typedef signed char schar;

void solve_l2r_lr_dual(_float** data,int num_trn, int num_ft,  int* y, double *w, double eps,	double Cp, double Cn, int classifier_maxiter)
{
	int l = num_trn;
	int w_size = num_ft;
	int i, s, iter = 0;

	double *xTx = new double[l];
	int max_iter = classifier_maxiter;
	int *index = new int[l];	
	double *alpha = new double[2*l]; // store alpha and C - alpha
	int max_inner_iter = 100; // for inner Newton
	double innereps = 1e-2;
	double innereps_min = min(1e-8, (double)eps);
	double upper_bound[3] = {Cn, 0, Cp};

	// Initial alpha can be set here. Note that
	// 0 < alpha[i] < upper_bound[GETI(i)]
	// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;

	for(i=0; i<l; i++)
	{
		xTx[i] = sparse_operator::nrm2_sq( num_ft, data[i] );
		sparse_operator::axpy(y[i]*alpha[2*i], num_ft, data[i], w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i=0; i<l; i++)
		{
			int j = i + get_rand_num( l-i );
			swap(index[i], index[j]);
		}

		int newton_iter = 0;
		double Gmax = 0;
		for (s=0; s<l; s++)
		{
			i = index[s];
			const _int yi = y[i];
			double C = upper_bound[GETI(i)];
			double ywTx = 0, xisq = xTx[i];
			ywTx = yi*sparse_operator::dot( w, num_ft, data[i] );
			double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
			if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
			{
				ind1 = 2*i+1;
				ind2 = 2*i;
				sign = -1;
			}

			//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
			double alpha_old = alpha[ind1];
			double z = alpha_old;
			if(C - z < 0.5 * C)
				z = 0.1*z;
			double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const _double eta = 0.1; // xi in the paper
			int inner_iter = 0;
			while (inner_iter <= max_inner_iter)
			{
				if(fabs(gp) < innereps)
					break;
				double gpp = a + C/(C-z)/z;
				double tmpz = z - gp/gpp;
				if(tmpz <= 0)
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				newton_iter++;
				inner_iter++;
			}

			if(inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C-z;
				sparse_operator::axpy(sign*(z-alpha_old)*yi, num_ft, data[i], w);
			}
		}

		iter++;

		if(Gmax < eps)
			break;

		if(newton_iter <= l/10)
			innereps = max(innereps_min, 0.1*innereps);

	}

	delete [] xTx;
	delete [] alpha;
	delete [] index;
}


void solve_l2r_l1l2_svc(_float** data,int num_trn, int num_ft,  int* y, double *w, double eps,	double Cp, double Cn, int siter)
{
	int l = num_trn;
	int w_size = num_ft;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = siter;
	int *index = new int[l];
	double *alpha = new double[l];
	int active_size = l;

	int tot_iter = 0;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};

	//d = pwd;
	//Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]

	
	for(i=0; i<l; i++)
		alpha[i] = 0;
	

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		//feature_node * const xi = prob->x[i];
		QD[i] += sparse_operator::nrm2_sq( num_ft, data[i] );
		sparse_operator::axpy(y[i]*alpha[i], num_ft, data[i], w);

		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			tot_iter ++;

			i = index[s];
			const int yi = y[i];
			//feature_node * const xi = prob->x[i];

			G = yi*sparse_operator::dot( w, num_ft, data[i] )-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			//cout << "update: " << i << " " << fabs(PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				//cout << " " << d;

				//print_nnz( w_size, w );

				sparse_operator::axpy(d, num_ft, data[i], w);

				//print_nnz( w_size, w );
			}
			//cout << endl;
		}

		iter++;

		/*
		if(iter % 10 == 0)
			cout << "." << endl;
		*/

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				//cout << "*" << endl;
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	//cout << " " << iter << " " << tot_iter ;

	// calculate objective value

	delete [] QD;
	delete [] alpha;
	delete [] index;
}
	

DMatF* compute_mu_plus(DMatF* trn_ft_mat, SMatF* trn_lbl_mat)
{
	SMatF* trn_lbl_mat_trans = trn_lbl_mat->transpose();
	DMatF* mu_plus = trn_ft_mat->prod(trn_lbl_mat_trans);
	mu_plus->unit_normalize_columns();
	return mu_plus;
}

void train_slice_generative_model(DMatF* trn_ft_mat, SMatF* trn_lbl_mat, string model_dir, string temp_dir, Param& params)
{
	DMatF* mu_plus = compute_mu_plus(trn_ft_mat, trn_lbl_mat);
	string mu_file = temp_dir+"/mu_plus.bin";
	mu_plus->write(mu_file, 1);

	string command = "python ANNS/train_hnsw.py ";
	string arguments =  mu_file + " " + model_dir + "/anns_model " + to_string(params.M) + " " + to_string(params.efC) + " " + to_string(params.num_threads) + " " + to_string(params.num_ft) + " cosinesimil";
	command += arguments;
	system(command.c_str());
}

IMat* read_multiple_imat_files(string file_dir, int num_files, bool input_format_is_binary)
{
	IMat* imat = new IMat(file_dir+"/0", input_format_is_binary);
	for(int i=1; i<num_files; i++)
	{
		IMat* tmat = new IMat(file_dir+"/"+to_string(i), input_format_is_binary);
		imat->append_mat_columnwise(tmat);
	}
	return imat;
}

IMat* find_most_confusing_negatives(DMatF* trn_ft_mat, string model_dir, string temp_dir, Param& params, float& io_time)
{
	Timer timer;
	timer.start();
	string trn_ft_file = temp_dir+"/trn_ft_mat.bin";
	trn_ft_mat->write(trn_ft_file, 1);
	timer.stop();
	string command = "python ANNS/test_hnsw.py ";
	string arguments = trn_ft_file + " " + model_dir + "/anns_model " + to_string(params.num_ft) + " " + to_string(params.num_lbl) + " " + to_string(params.efS) + " " + to_string(params.num_nbrs) + " 0 " + temp_dir + " " + to_string(params.num_threads) + " " + to_string(params.num_io_threads) + " cosinesimil";
	command += arguments;
  system(command.c_str());
	timer.resume();
	IMat* temp_imat = read_multiple_imat_files(temp_dir, params.num_io_threads, 0);
	IMat* trn_negatives = temp_imat->transpose();
	io_time = timer.stop();
	delete temp_imat;
	return trn_negatives;
}	

DMatF* train_discriminative_classifier(DMatF* trn_ft_mat, SMatF* trn_lbl_mat, IMat* trn_negatives, Param& params, string model_dir)
{
    if (params.debug) {
        Logger::log("params.generate_explanations = " + std::to_string(params.generate_explanations));
        Logger::log("params.adversarial_training = " + std::to_string(params.adversarial_training));
        Logger::log("params.num_features_to_explain = " + std::to_string(params.num_features_to_explain));
        Logger::log("params.perturbation_strength = " + std::to_string(params.perturbation_strength));
    }

	SMatF* trn_lbl_mat_trans = trn_lbl_mat->transpose();
	trn_ft_mat->unit_normalize_columns();
	Logger::log("Training discriminative classifier...");
	_int num_ft = trn_ft_mat->nr;
	for(_int i=0; i<trn_ft_mat->nc; i++)
	{
		Realloc(num_ft,num_ft+1,trn_ft_mat->data[i]);
		trn_ft_mat->data[i][num_ft] = 1;
	}
	trn_ft_mat->nr = num_ft+1;	
	_int num_trn = trn_ft_mat->nc;
	num_ft = trn_ft_mat->nr;
	_int num_lbl = trn_lbl_mat_trans->nc;

	Logger::log("Number of training examples: " + std::to_string(num_trn));
	Logger::log("Number of features: " + std::to_string(num_ft));
	float th = params.classifier_threshold;
	double eps = 0.1;
	double Cp = params.classifier_cost;
	double Cn = params.classifier_cost; 

	DMatF* w_dis = new DMatF( num_ft, num_lbl );

	// updated code start
    DMatF* feature_importance = new DMatF(num_ft, num_lbl);
	Logger::log("Computing feature importance...");
	// If adversarial training is enabled, generate perturbed examples
    DMatF* adv_trn_ft_mat = NULL;
    if (params.adversarial_training) {
		Logger::log("Adversarial training enabled.");
		Logger::log("Generating perturbed training examples...");
        // Create a copy of the training features for perturbation
        adv_trn_ft_mat = new DMatF();
        adv_trn_ft_mat->nr = trn_ft_mat->nr;
        adv_trn_ft_mat->nc = trn_ft_mat->nc;
        adv_trn_ft_mat->data = new _float*[adv_trn_ft_mat->nc];
        
        // Generate slightly perturbed versions of training examples
        mt19937 gen(42); // Fixed seed for reproducibility
        normal_distribution<_float> dist(0, params.perturbation_strength);
        
        for (_int i = 0; i < trn_ft_mat->nc; i++) {
            adv_trn_ft_mat->data[i] = new _float[adv_trn_ft_mat->nr];
            for (_int j = 0; j < trn_ft_mat->nr; j++) {
                // Add random noise
                adv_trn_ft_mat->data[i][j] = trn_ft_mat->data[i][j] + dist(gen);
                // Ensure non-negative values
                if (adv_trn_ft_mat->data[i][j] < 0)
                    adv_trn_ft_mat->data[i][j] = 0;
            }
        }
    }
	// updated code end

	omp_set_dynamic(0);
	omp_set_num_threads(params.num_threads);
	#pragma omp parallel shared(trn_ft_mat,trn_lbl_mat_trans,trn_negatives, w_dis, num_ft, num_lbl, th, eps, Cp, Cn)
	{
	#pragma omp for
	for( int l=0; l<num_lbl; l++)
	{
		Logger::threadLog("Training into parallel omp  " + std::to_string(l));
		_int sl_size = 0;
		VecI positives(num_trn, 0);
		for (int i=0; i<trn_lbl_mat_trans->size[l]; i++)
			positives[trn_lbl_mat_trans->data[l][i].first] = 1;

		int overlap = 0;
		for (int i=0; i<trn_negatives->size[l]; i++)
		{
			if (positives[trn_negatives->data[l][i]]>0)
			{
				positives[trn_negatives->data[l][i]] = -1;
				overlap++;
			}
		}
		sl_size = trn_negatives->size[l]+trn_lbl_mat_trans->size[l]-overlap;

		if (sl_size==0)
		{
			for( _int f=0; f<num_ft; f++ )
				w_dis->data[l][f] = 0;
			continue;
		}
		_int* y = new _int[sl_size];
		_float** data = new _float*[sl_size];
		_int inst;
		for (int i=0; i<trn_negatives->size[l]; i++)
		{
			inst = trn_negatives->data[l][i];
			data[i] = trn_ft_mat->data[inst];
			if (positives[inst]!=0)
				y[i] = +1;
			else
				y[i] = -1;
		}
		int ctr = trn_negatives->size[l];
		for (int i=0; i<trn_lbl_mat_trans->size[l]; i++)
		{
			inst = trn_lbl_mat_trans->data[l][i].first;
			if (positives[inst]>0)
			{
				data[ctr] = trn_ft_mat->data[inst];
				y[ctr] = +1;
				ctr++;
			}
		}
		// Logger::threadLog("Training label " + std::to_string(l) + " with " + std::to_string(sl_size) + " examples.");
		// Logger::threadLog("before adversarial training");
		//updated code start
		bool arrays_replaced = false;
		_int* training_y = nullptr;
		_float** training_data = nullptr;
		_int training_size = 0;
		if (params.adversarial_training && adv_trn_ft_mat != NULL) {
			_int orig_sl_size = sl_size;
			// Logger::threadLog("inside adversarial training");

			// Count positive examples first
			_int pos_count = 0;
			for (_int idx = 0; idx < sl_size; idx++) {
				if (y[idx] == 1) pos_count++;
			}
			_int potential_adv_count = pos_count;

			// Expand arrays to include adversarial examples
			_int* new_y = new _int[sl_size + potential_adv_count];
			_float** new_data = new _float*[sl_size + potential_adv_count];
			// Logger::threadLog("New size: " + std::to_string(sl_size * 2));
			// Copy original data
			for (_int idx = 0; idx < sl_size; idx++) {
				new_y[idx] = y[idx];
				new_data[idx] = data[idx];
			}
			
			// Add adversarial versions of positive examples
			_int adv_count = 0;
			Logger::threadLog("Adding adversarial examples..." + std::to_string(potential_adv_count));	
			for (_int idx = 0; idx < sl_size; idx++) {
				if (y[idx] == 1) {  // Only perturb positive examples
					_int inst_idx = -1;
					// Logger::threadLog("Finding adversarial example for index " + std::to_string(idx));
					// Find the original instance index
					for (_int j = 0; j < trn_negatives->size[l]; j++) {
						if (data[idx] == trn_ft_mat->data[trn_negatives->data[l][j]]) {
							inst_idx = trn_negatives->data[l][j];
							break;
						}
					}
					
					if (inst_idx == -1) {
						for (_int j = 0; j < trn_lbl_mat_trans->size[l]; j++) {
							if (data[idx] == trn_ft_mat->data[trn_lbl_mat_trans->data[l][j].first]) {
								inst_idx = trn_lbl_mat_trans->data[l][j].first;
								break;
							}
						}
					}
					
					if (inst_idx != -1) {
						new_data[sl_size + adv_count] = adv_trn_ft_mat->data[inst_idx];
						new_y[sl_size + adv_count] = y[idx];  // Same label as original
						adv_count++;
					}
				}
			}

			// Logger::threadLog("Creating adversarial examples: " + std::to_string(adv_count) + " found.");
			// Logger::threadLog("New training size: " + std::to_string(sl_size) + " -> " + std::to_string(sl_size + adv_count));
			
			if (adv_count > 0) {
				// Update the total size to include adversarial examples
				_int new_sl_size = sl_size + adv_count;
				Logger::threadLog("New training size: " + std::to_string(new_sl_size));
				// FREE THE ORIGINAL ARRAYS
				// _int* old_y = y;
				// _float** old_data = data;
				
				// // SWITCH TO THE NEW ARRAYS BEFORE TRAINING
				// y = new_y;
				// data = new_data;
				// sl_size = new_sl_size;
				
				// // Set flag to indicate arrays were replaced
				// arrays_replaced = true;
				
				// // Delete original arrays now that we've switched
				// delete[] old_y;
				// delete[] old_data;
				training_y = new_y;
				training_data = new_data;
				training_size = new_sl_size;
				arrays_replaced = true;

			} else {
				// No adversarial examples were created, so we can just use the original arrays
				Logger::threadLog("No adversarial examples created.");
				delete[] new_y;
				delete[] new_data;

				training_y = y;
				training_data = data;
				training_size = sl_size;
			}
		} else{
			// No adversarial training, use original arrays
			training_y = y;
			training_data = data;
			training_size = sl_size;
		}
		
		Logger::threadLog("Training l  " + std::to_string(l));
		double* w = new double[ num_ft ]();		
		if(params.classifier_kind==0)
			solve_l2r_l1l2_svc( data, sl_size, num_ft, y, w, eps, Cp, Cn, params.classifier_maxiter);
		else
			solve_l2r_lr_dual(data, sl_size, num_ft, y, w, eps, Cp, Cn, params.classifier_maxiter);

		for( _int f=0; f<num_ft; f++ )
		{
			if( fabs( w[f] ) > th )
			{
				w_dis->data[l][f] = w[f];
			}
			else w_dis->data[l][f] = 0;

			feature_importance->data[l][f] = fabs(w_dis->data[l][f]);
		}


		Logger::threadLog("Finished training label " + std::to_string(l) + ". Found " + 
			std::to_string(std::count_if(w_dis->data[l], w_dis->data[l] + num_ft, 
									[&](float f) { return fabs(f) > th; })) + 
			" non-zero weights.");
		if (w) delete [] w;

		if (arrays_replaced) {
			if (training_y) delete[] training_y;
			if (training_data) delete[] training_data;
		} else {
			if (y) delete[] y;
			if (data) delete[] data;
		}
		Logger::threadLog("Finished training label " + std::to_string(l) + " with " + std::to_string(sl_size) + " examples.");
	}
	}
		
	//updated code start
	// if (params.adversarial_training && adv_trn_ft_mat != NULL) {
	// 	for (_int i = 0; i < adv_trn_ft_mat->nc; i++) {
	// 		delete[] adv_trn_ft_mat->data[i];
	// 	}
	// 	delete[] adv_trn_ft_mat->data;
	// 	delete adv_trn_ft_mat;
	// }
	Logger::log("Saving feature importance...");
	feature_importance->write(model_dir + "/feature_importance.txt", 0);
	Logger::log("Feature importance saved to " + model_dir + "/feature_importance.txt");
	delete feature_importance;
	//updated code end
	delete trn_lbl_mat_trans;
	return w_dis;
}

DMatF* train_slice(DMatF* trn_ft_mat, SMatF* trn_lbl_mat, string model_dir, Param& params, float& train_time)
{
	float io_time = 0.0;
	float* t_time = new float;
	*t_time = 0;
	Timer timer;

	string temp_dir = model_dir + "/tmp";
	mkdir(temp_dir.c_str(), S_IRWXU);
	timer.start();

	if (!params.quiet)
		printf("Training generative model ...\n");	
	train_slice_generative_model(trn_ft_mat, trn_lbl_mat, model_dir, temp_dir, params);

	if (!params.quiet)
		printf("Finding the most confusing negatives ...\n");
	IMat* trn_negatives = find_most_confusing_negatives(trn_ft_mat, model_dir, temp_dir, params, io_time);

	if (!params.quiet)
		printf("Training discriminative classifiers ...\n");
	DMatF* w_discriminative = train_discriminative_classifier(trn_ft_mat, trn_lbl_mat, trn_negatives, params, model_dir);

	*t_time += timer.stop();
	train_time = *t_time;
	train_time -= io_time;
	delete t_time;
	delete trn_negatives;
	return w_discriminative;
}

SMatF* evaluate_discriminative_model(DMatF* tst_ft_mat, DMatF* w_dis, SMatF* shortlist, Param& params, int K)
{
	_int num_ft = tst_ft_mat->nr;
	_int num_lbl = w_dis->nc;
	_int num_tst = tst_ft_mat->nc;

	float gamma = 0;
	for (int i=0;i<num_lbl;i++)
	{
		float temp = 0;	
		for (int j=0;j<num_ft;j++)
			temp += pow(w_dis->data[i][j],2.0);
		gamma += sqrt(temp);
	}
	gamma = gamma/float(num_lbl);

	SMatF* score_mat = new SMatF();	
	score_mat->nr = num_lbl;
	score_mat->nc = num_tst;
	score_mat->size = new _int[num_tst]();
	score_mat->data = new pairIF*[num_tst];

	omp_set_dynamic(0);
	omp_set_num_threads(params.num_threads);
	#pragma omp parallel shared(tst_ft_mat,w_dis,shortlist,score_mat,num_ft, num_tst)
	{
	#pragma omp for
	for(_int i=0; i<num_tst; i++)
	{
		//if ((i%1000)==0)
		//	printf("%d\n",i);
		
		score_mat->data[i] = new pairIF[shortlist->size[i]];
		_int ctr = 0;
		_float max1 = 0;
		_float max_dist = 0;
		for(_int j=0; j<shortlist->size[i]; j++)
		{
			_int ind = shortlist->data[i][j].first;
			_float prod = 0;
			_int ctr1 = 0;
			_int ctr2 = 0;
			for(_int f=0; f<num_ft; f++)
				prod += w_dis->data[ind][f]*tst_ft_mat->data[i][f];
			score_mat->data[i][j].first = ind;
			score_mat->data[i][j].second = prod;
		}
		for(_int j=0; j<shortlist->size[i]; j++)
			score_mat->data[i][j].second = (1.0/(1.0+exp(-score_mat->data[i][j].second))) + (1.0/(1.0 + exp(-shortlist->data[i][j].second*gamma + params.b_gen)));
		pairIF* vec = score_mat->data[i];
		sort(vec, vec+shortlist->size[i], comp_pair_by_second_desc<_int,_float>);
		_int k = K;
		if (k>shortlist->size[i])
			k = shortlist->size[i];
		Realloc(shortlist->size[i],k,score_mat->data[i]);
		vec = score_mat->data[i];
		sort(vec, vec+k, comp_pair_by_first<_int,_float>);
		score_mat->size[i] = k;
	}
	}
	return score_mat;
}

void generate_prediction_explanation(DMatF* tst_ft_mat, DMatF* w_dis, SMatF* score_mat, 
	DMatF* feature_importance, string explanation_file,
	int num_features_to_explain = 5) {
	ofstream fout(explanation_file);

	// Write header
	fout << "instance_id,label_id,score,feature_id,feature_importance,contribution" << endl;

	// For each test instance
	for (_int i = 0; i < score_mat->nc; i++) {
		// For each predicted label
		for (_int j = 0; j < score_mat->size[i]; j++) {
			_int label_id = score_mat->data[i][j].first;
			_float score = score_mat->data[i][j].second;

			// Create a vector of (feature_id, importance, contribution) tuples
			vector<tuple<_int, _float, _float>> feature_contributions;

			// Calculate contribution of each feature
			for (_int f = 0; f < tst_ft_mat->nr; f++) {
				_float importance = feature_importance->data[label_id][f];
				_float contribution = importance * tst_ft_mat->data[i][f] * w_dis->data[label_id][f];

				if (fabs(importance) > 1e-6) { // Only consider important features
					feature_contributions.push_back(make_tuple(f, importance, contribution));
				}
			}
			// Sort by absolute contribution
			sort(feature_contributions.begin(), feature_contributions.end(), 
			[](const tuple<_int, _float, _float>& a, const tuple<_int, _float, _float>& b) {
				return fabs(get<2>(a)) > fabs(get<2>(b));
			});

			// Output top contributing features
			int num_to_show = min(num_features_to_explain, (int)feature_contributions.size());
			for (int k = 0; k < num_to_show; k++) {
				fout << i << "," << label_id << "," << score << "," 
				<< get<0>(feature_contributions[k]) << "," 
				<< get<1>(feature_contributions[k]) << "," 
				<< get<2>(feature_contributions[k]) << endl;
			}
		}
	}
	fout.close();

	vector<_float> global_feature_importance(tst_ft_mat->nr, 0);

	for (_int f = 0; f < tst_ft_mat->nr; f++) {
		for (_int l = 0; l < w_dis->nc; l++) {
			global_feature_importance[f] += fabs(feature_importance->data[l][f]);
		}
	}

	// Write global importance to a separate file
	string global_importance_file = explanation_file + ".global";
	ofstream global_out(global_importance_file);
	global_out << "feature_id,global_importance" << endl;
	for (_int f = 0; f < tst_ft_mat->nr; f++) {
		global_out << f << "," << global_feature_importance[f] << endl;
	}
	global_out.close();
}

SMatF* read_multiple_smat_files(string file_dir, int num_files, bool input_format_is_binary)
{
	SMatF* smat = new SMatF(file_dir+"/0", input_format_is_binary);
	for(int i=1; i<num_files; i++)
	{
		SMatF* tmat = new SMatF(file_dir+"/"+to_string(i), input_format_is_binary);
		smat->append_mat_columnwise(tmat);
	}
	return smat;
}


SMatF* evaluate_generative_model(DMatF* tst_ft_mat, string model_dir, string temp_dir, Param& params, float& io_time)
{
	Timer timer;
	timer.start();
	string tst_ft_file = temp_dir+"/tst_ft_mat.bin";
	tst_ft_mat->write(tst_ft_file, 1);
	timer.stop();
	string command = "python ANNS/test_hnsw.py ";
	string arguments = tst_ft_file + " " + model_dir + "/anns_model " + to_string(params.num_ft) + " " + to_string(params.num_lbl) + " " + to_string(params.efS) + " " + to_string(params.num_nbrs) + " 1 " + temp_dir + " " + to_string(params.num_threads) + " " + to_string(params.num_io_threads) + " cosinesimil";
	command += arguments;
  system(command.c_str());
	timer.resume();
	SMatF* generative_score_mat = read_multiple_smat_files(temp_dir, params.num_io_threads, 0);
	io_time = timer.stop();
	return generative_score_mat;
}	

SMatF* predict_slice(DMatF* tst_ft_mat, DMatF* w_dis, string model_dir, Param& params, float& test_time) 
{
	float io_time = 0.0;
	float* t_time = new float;
	*t_time = 0;
	Timer timer;

	string temp_dir = model_dir + "/tmp";
	mkdir(temp_dir.c_str(), S_IRWXU);

	timer.start();
	tst_ft_mat->unit_normalize_columns();

	// updated code start
	if (params.adversarial_defense) {
        for (_int i = 0; i < tst_ft_mat->nc; i++) {
            // Check for unusual feature patterns
            _float feature_mean = 0;
            _float feature_std = 0;
            
            // Calculate mean and std of features
            for (_int f = 0; f < tst_ft_mat->nr; f++) {
                feature_mean += tst_ft_mat->data[i][f];
            }
            feature_mean /= tst_ft_mat->nr;
            
            for (_int f = 0; f < tst_ft_mat->nr; f++) {
                feature_std += pow(tst_ft_mat->data[i][f] - feature_mean, 2);
            }
            feature_std = sqrt(feature_std / tst_ft_mat->nr);
            
            // Identify and clip extreme values (potential adversarial perturbations)
            _float clip_threshold = feature_mean + 3 * feature_std; // 3 sigma rule
            
            for (_int f = 0; f < tst_ft_mat->nr; f++) {
                if (tst_ft_mat->data[i][f] > clip_threshold) {
                    // Clip extreme values
                    tst_ft_mat->data[i][f] = clip_threshold;
                }
            }
            
            // Re-normalize after clipping
            _float norm = 0;
            for (_int f = 0; f < tst_ft_mat->nr; f++) {
                norm += tst_ft_mat->data[i][f] * tst_ft_mat->data[i][f];
            }
            norm = sqrt(norm);
            
            if (norm > 0) {
                for (_int f = 0; f < tst_ft_mat->nr; f++) {
                    tst_ft_mat->data[i][f] /= norm;
                }
            }
        }
    }
	// updated code end

	SMatF* score_mat_gen = evaluate_generative_model(tst_ft_mat, model_dir, temp_dir, params, io_time);
	
	_int num_ft = tst_ft_mat->nr;
	for(_int i=0; i<tst_ft_mat->nc; i++)
	{
		Realloc(num_ft,num_ft+1,tst_ft_mat->data[i]);
		tst_ft_mat->data[i][num_ft] = 1;
	}
	tst_ft_mat->nr = num_ft+1;
	
	SMatF* score_mat = evaluate_discriminative_model(tst_ft_mat, w_dis, score_mat_gen, params, 20);
	*t_time += timer.stop();
	test_time = *t_time;
	test_time -= io_time;

	delete score_mat_gen;

	// updated code start
    if (params.generate_explanations) {
        // Load feature importance
        DMatF* feature_importance = new DMatF(model_dir + "/feature_importance.txt", 0);
        
		string explanation_file = "./Sandbox/Results/EURLex-4K/explanations/explanations.csv";
		printf("Generating explanations to: %s\n", explanation_file.c_str());
		fflush(stdout);
		
        // Generate explanations
        generate_prediction_explanation(tst_ft_mat, w_dis, score_mat, 
                                       feature_importance, 
                                      explanation_file,
									params.num_features_to_explain);      
        delete feature_importance;
    }	
	// updated code end 


	return score_mat;
}
