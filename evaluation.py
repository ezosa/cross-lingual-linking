from doc_linking import ensemble_dist_ranker, ensemble_dist_ranker_weighted, \
	compute_precision_at_n, compute_mrr, get_normalized_distance, get_mean_centered_distance
import pickle
import numpy as np
from scipy.stats import rankdata
from scipy.stats import spearmanr

# Multilingual topic models trained using the MLTM script
tm_model_names = ["yle_100topics_2012", "yle_100topics_2013", "yle_100topics_2014"]

test_years = [2013, 2014, 2015]
precision_k = [5,10]


def compute_mean_correlation(ranks1, ranks2):
	mean_corr = []
	for i in range(len(ranks1)):
		#print(i)
		r1 = ranks1[i]
		r2 = ranks2[i]
		r = spearmanr(r1, r2)[0]
		if not np.isnan(r):
			mean_corr.append(r)
	mean_corr = np.mean(mean_corr)
	return mean_corr


for i, test_year in enumerate(test_years):
	print("Test year:", test_year)
	tm_model_name = tm_model_names[i]
	mrr_tm_months = []
	nap_tm_months = []
	prec_tm_months = {k: [] for k in precision_k}

	mrr_cr5_months = []
	nap_cr5_months = []
	prec_cr5_months = {k: [] for k in precision_k}

	mrr_wass_months = []
	nap_wass_months = []
	prec_wass_months = {k: [] for k in precision_k}

	mrr_wass_months2 = []
	prec_wass_months2 = {k: [] for k in precision_k}

	mrr_e1_months = []
	nap_e1_months = []
	prec_e1_months = {k: [] for k in precision_k}

	mrr_e2_months = []
	nap_e2_months = []
	prec_e2_months = {k: [] for k in precision_k}

	mrr_e3_months = []
	nap_e3_months = []
	prec_e3_months = {k: [] for k in precision_k}

	mrr_e4_months = []
	nap_e4_months = []
	prec_e4_months = {k: [] for k in precision_k}

	mrr_e5_months = []
	nap_e5_months = []
	prec_e5_months = {k: [] for k in precision_k}

	mrr_e6_months = []
	nap_e6_months = []
	prec_e6_months = {k: [] for k in precision_k}

	mrr_e7_months = []
	nap_e7_months = []
	prec_e7_months = {k: [] for k in precision_k}

	mrr_e8_months = []
	nap_e8_months = []
	prec_e8_months = {k: [] for k in precision_k}

	tm_cr5_corr_year = []
	tm_wass_corr_year = []
	cr5_wass_corr_year = []

	for test_month in [1,2,3,4,5,6,7,8,9,10,11,12]:
		#print("Test month:", test_month)
		tm_results_file = "results/" + tm_model_name + "_" + str(test_year) + "_" + str(test_month) + "_iter.pkl"
		cr5_results_file = "results/cr5_" + str(test_year) + "_" + str(test_month) + ".pkl"
		wass_results_file = "results/wass_" + str(test_year) + "_" + str(test_month) + ".pkl"

		tm_results = pickle.load(open(tm_results_file, "rb"))
		cr5_results = pickle.load(open(cr5_results_file, "rb"))
		wass_results = pickle.load(open(wass_results_file, "rb"))

		ground_truth = tm_results['ground_truth']
		sv_keys_dict = tm_results['sv_keys_dict']
		wass_ground_truth = wass_results['ground_truth']

		tm_dist = tm_results['distances']
		tm_pred = tm_results['predictions']
		tm_ranks = [rankdata(tm_dist[i], method='ordinal') for i in range(tm_dist.shape[0])]
		cr5_dist = cr5_results['distances']
		cr5_pred = cr5_results['predictions']
		cr5_ranks = [rankdata(cr5_dist[i], method='ordinal') for i in range(cr5_dist.shape[0])]
		for i in range(len(cr5_pred)):
			for j in range(len(cr5_pred[i])):
				if cr5_pred[i][j] in sv_keys_dict:
					cr5_pred[i][j] = sv_keys_dict[cr5_pred[i][j]]
				else:
					cr5_pred[i][j] = -1

		wass_dist = wass_results['distances']
		wass_pred = wass_results['predictions']
		wass_ranks = [rankdata(wass_dist[i], method='ordinal') for i in range(wass_dist.shape[0])]

		tm_cr5_corr = compute_mean_correlation(tm_ranks, cr5_ranks)
		tm_wass_corr = compute_mean_correlation(tm_ranks, wass_ranks)
		cr5_wass_corr = compute_mean_correlation(cr5_ranks, wass_ranks)

		tm_cr5_corr_year.append(tm_cr5_corr)
		tm_wass_corr_year.append(tm_wass_corr)
		cr5_wass_corr_year.append(cr5_wass_corr)
		print("MLTM & Cr5 mean r:", tm_cr5_corr)
		print("MLTM & Wass mean r:", tm_wass_corr)
		print("Cr5 & Wass mean r:", cr5_wass_corr)

	tm_cr5_corr_year = np.mean(tm_cr5_corr_year)
	tm_wass_corr_year = np.mean(tm_wass_corr_year)
	cr5_wass_corr_year = np.mean(cr5_wass_corr_year)
	print("MLTM & Cr5 mean r:", tm_cr5_corr_year)
	print("MLTM & Wass mean r:", tm_wass_corr_year)
	print("Cr5 & Wass mean r:", cr5_wass_corr_year)

		tm_dist_normal = get_normalized_distance(tm_dist)
		cr5_dist_normal = get_normalized_distance(cr5_dist)
		wass_dist_normal = get_normalized_distance(wass_dist)

		tm_dist_centered = get_mean_centered_distance(tm_dist)
		cr5_dist_centered = get_mean_centered_distance(cr5_dist)
		wass_dist_centered = get_mean_centered_distance(wass_dist)

		mltm_cr5_wass_dist_pred = ensemble_dist_ranker(dist1=tm_dist_centered, dist2=cr5_dist_centered, dist3=wass_dist_centered, ensemble_method="mean")
		mltm_cr5_dist_pred = ensemble_dist_ranker(dist1=tm_dist_centered, dist2=cr5_dist_centered, ensemble_method="mean")
		cr5_wass_dist_pred = ensemble_dist_ranker(dist1=cr5_dist_centered, dist2=wass_dist_centered, ensemble_method="mean")
		mltm_wass_dist_pred = ensemble_dist_ranker(dist1=tm_dist_centered, dist2=wass_dist_centered, ensemble_method="mean")

		mltm_cr5_dist_pred = ensemble_dist_ranker_weighted(dist1=tm_dist_centered, dist2=cr5_dist_centered)
		cr5_wass_dist_pred = ensemble_dist_ranker_weighted(dist1=cr5_dist_centered, dist2=wass_dist_centered)
		mltm_wass_dist_pred = ensemble_dist_ranker_weighted(dist1=tm_dist_centered, dist2=wass_dist_centered)

		# compute MRR for TM, Cr5, Wasserstein, Ensemble1, Ensemble2
		# stand-alone models
		mrr_tm = compute_mrr(tm_pred, ground_truth)
		mrr_tm_months.append(mrr_tm)
		mrr_cr5 = compute_mrr(cr5_pred, ground_truth)
		mrr_cr5_months.append(mrr_cr5)
		mrr_wass = compute_mrr(wass_pred, wass_ground_truth)
		mrr_wass_months.append(mrr_wass)
		# ensemble models
		mrr_e1 = compute_mrr(mltm_cr5_wass_dist_pred, ground_truth)
		mrr_e1_months.append(mrr_e1)
		mrr_e2 = compute_mrr(mltm_cr5_dist_pred, ground_truth)
		mrr_e2_months.append(mrr_e2)
		mrr_e3 = compute_mrr(cr5_wass_dist_pred, ground_truth)
		mrr_e3_months.append(mrr_e3)
		mrr_e4 = compute_mrr(mltm_wass_dist_pred, ground_truth)
		mrr_e4_months.append(mrr_e4)
		# mrr_e5 = compute_mrr2(mltm_cr5_wass_rank_pred, ground_truth)
		# mrr_e5_months.append(mrr_e5)
		# mrr_e6 = compute_mrr2(mltm_cr5_rank_pred, ground_truth)
		# mrr_e6_months.append(mrr_e6)
		# mrr_e7 = compute_mrr2(cr5_wass_rank_pred, ground_truth)
		# mrr_e7_months.append(mrr_e7)
		# mrr_e8 = compute_mrr2(mltm_wass_rank_pred, ground_truth)
		# mrr_e8_months.append(mrr_e8)

		# compute precision for TM, Cr5, Wasserstein
		for k in precision_k:
			#print("***** Topic Model *****")
			prec_tm = compute_precision_at_n(tm_pred, ground_truth, k)
			prec_tm_months[k].append(prec_tm)
			#print("***** Cr5 *****")
			prec_cr5 = compute_precision_at_n(cr5_pred, ground_truth, k)
			prec_cr5_months[k].append(prec_cr5)
			#print("***** Wasserstein *****")
			prec_wass = compute_precision_at_n(wass_pred, wass_ground_truth, k)
			prec_wass_months[k].append(prec_wass)
			prec_e1 = compute_precision_at_n(mltm_cr5_wass_dist_pred, ground_truth, k)
			prec_e1_months[k].append(prec_e1)
			prec_e2 = compute_precision_at_n(mltm_cr5_dist_pred, ground_truth, k)
			prec_e2_months[k].append(prec_e2)
			prec_e3 = compute_precision_at_n(cr5_wass_dist_pred, ground_truth, k)
			prec_e3_months[k].append(prec_e3)
			prec_e4 = compute_precision_at_n(mltm_wass_dist_pred, ground_truth, k)
			prec_e4_months[k].append(prec_e4)
			# prec_e5 = compute_precision_at_n(mltm_cr5_wass_rank_pred, ground_truth, k)
			# prec_e5_months[k].append(prec_e5)
			# prec_e6 = compute_precision_at_n(mltm_cr5_rank_pred, ground_truth, k)
			# prec_e6_months[k].append(prec_e6)
			# prec_e7 = compute_precision_at_n(cr5_wass_rank_pred, ground_truth, k)
			# prec_e7_months[k].append(prec_e7)
			# prec_e8 = compute_precision_at_n(mltm_wass_rank_pred, ground_truth, k)
			# prec_e8_months[k].append(prec_e8)

	mrr_tm_mean = np.mean(mrr_tm_months)
	mrr_cr5_mean = np.mean(mrr_cr5_months)
	mrr_wass_mean = np.mean(mrr_wass_months)

	mrr_e1_mean = np.mean(mrr_e1_months)
	mrr_e2_mean = np.mean(mrr_e2_months)
	mrr_e3_mean = np.mean(mrr_e3_months)
	mrr_e4_mean = np.mean(mrr_e4_months)
	# mrr_e5_mean = np.mean(mrr_e5_months)
	# mrr_e6_mean = np.mean(mrr_e6_months)
	# mrr_e7_mean = np.mean(mrr_e7_months)
	# mrr_e8_mean = np.mean(mrr_e8_months)

	# nap_tm_mean = np.mean(nap_tm_months)
	# nap_cr5_mean = np.mean(nap_cr5_months)
	# nap_wass_mean = np.mean(nap_wass_months)
	# nap_e1_mean = np.mean(nap_e1_months)
	# nap_e2_mean = np.mean(nap_e2_months)
	# nap_e3_mean = np.mean(nap_e3_months)
	# nap_e4_mean = np.mean(nap_e4_months)
	# nap_e5_mean = np.mean(nap_e5_months)
	# nap_e6_mean = np.mean(nap_e6_months)
	# nap_e7_mean = np.mean(nap_e7_months)

	print("\n=========== Test year:", test_year,"===========")
	print("***** MRR *****")
	print("TM:", mrr_tm_mean)
	print("Cr5:", mrr_cr5_mean)
	print("Wasserstein:", mrr_wass_mean)
	print("------")
	print("MTLM_Cr5_Wass:", mrr_e1_mean)
	print("MLTM_Cr5:", mrr_e2_mean)
	print("Cr5_Wass:", mrr_e3_mean)
	print("MLTM_Wass:", mrr_e4_mean)
	print("------")
	# print("MLTM_Cr5_Wass_Rank:", mrr_e5_mean)
	# print("MLTM_Cr5_Rank:", mrr_e6_mean)
	# print("Cr5_Wass_Rank:", mrr_e7_mean)
	# print("MLTM_Wass_Rank:", mrr_e8_mean)
	# print("\n***** Non-interpolated AP *****")
	# print("TM:", nap_tm_mean)
	# print("Cr5:", nap_cr5_mean)
	# print("Wasserstein:", nap_wass_mean)
	# print("MTLM_Cr5_Wass:", nap_e1_mean)
	# print("MLTM_Cr5:", nap_e2_mean)
	# print("Cr5_Wass:", nap_e3_mean)
	# print("MLTM_Wass:", nap_e4_mean)
	# print("Ensemble5:", nap_e5_mean)
	# print("Ensemble6:", nap_e6_mean)
	# print("Ensemble7:", nap_e7_mean)
	print("\n***** Precision @ k *****")
	for k in precision_k:
		prec_tm_mean = np.mean(prec_tm_months[k])
		prec_cr5_mean = np.mean(prec_cr5_months[k])
		prec_wass_mean = np.mean(prec_wass_months[k])
		prec_e1_mean = np.mean(prec_e1_months[k])
		prec_e2_mean = np.mean(prec_e2_months[k])
		prec_e3_mean = np.mean(prec_e3_months[k])
		prec_e4_mean = np.mean(prec_e4_months[k])
		# prec_e5_mean = np.mean(prec_e5_months[k])
		# prec_e6_mean = np.mean(prec_e6_months[k])
		# prec_e7_mean = np.mean(prec_e7_months[k])
		# prec_e8_mean = np.mean(prec_e8_months[k])
		print("k =", k)
		print("---------------")
		print("TM:", prec_tm_mean)
		print("Cr5:", prec_cr5_mean)
		print("Wasserstein:", prec_wass_mean)
		print("------")
		print("MTLM_Cr5_Wass:", prec_e1_mean)
		print("MLTM_Cr5:", prec_e2_mean)
		print("Cr5_Wass:", prec_e3_mean)
		print("MLTM_Wass:", prec_e4_mean)
		print("------")
		# print("MLTM_Cr5_Wass_Rank:", prec_e5_mean)
		# print("MLTM_Cr5_Rank:", prec_e6_mean)
		# print("Cr5_Wass_Rank:", prec_e7_mean)
		# print("MLTM_Wass_Rank:", prec_e8_mean)
		# print("---------------")

