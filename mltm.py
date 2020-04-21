import random
import os
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import tarfile
from datetime import date
import datetime
import calendar
import pickle
import re

stopwords_yle = set(stopwords.words('finnish')).union(stopwords.words('swedish'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def getKey(item):
    return item[1]


def compute_frequency_scores(documents):
	languages = list(documents.keys())
	scores = {}
	for lang in languages:
		articles = documents[lang]
		tokens = [token for art in articles for token in art]
		counts = Counter(tokens)
		tuples = [(key, counts[key]) for key in counts.keys()]
		sorted_tuples = sorted(tuples, key=getKey, reverse=True)
		scores[lang] = sorted_tuples
	return scores


def prune_vocabulary(documents, vocab_len=2000):
	print("Cutting the vocab to", vocab_len, "tokens")
	term_scores = compute_frequency_scores(documents)
	languages = list(documents.keys())
	dictionary = {lang: set() for lang in languages}
	for lang in languages:
			valid_tokens = [term[0] for term in term_scores[lang][:vocab_len]]
			n_docs = len(documents[lang])
			for d in range(n_docs):
				doc = documents[lang][d]
				pruned_doc = [w for w in doc if w in valid_tokens]
				documents[lang][d] = pruned_doc
				dictionary[lang].update(pruned_doc)
	for lang in languages:
		dictionary[lang] = list(dictionary[lang])
	return documents, dictionary


def clean_yle(doc):
	clean_short = " ".join([tok for tok in doc if len(tok)>2])
	clean_punc = ''.join(ch for ch in clean_short if ch not in exclude)
	clean_stop = [i for i in clean_punc.lower().split() if i not in stopwords_yle and len(i)>2 and 'http' not in i]
	clean_digits = " ".join([i for i in clean_stop if re.match(r'^([\s\d]+)$', i) is None])
	return clean_digits


def clean_yle_new_articles(doc, valid_words):
	clean_short = " ".join([tok for tok in doc if len(tok)>2])
	clean_punc = ''.join(ch for ch in clean_short if ch not in exclude)
	clean_stop = [i for i in clean_punc.lower().split() if i not in stopwords_yle]
	clean_others = [i for i in clean_stop if i in valid_words]
	return clean_others


def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day)


def get_yle_news_corpus(max_doc_per_timeslice=0, n_timeslices=1, ts=0, start_year=2012):
	print("getting YLE data")
	yle_filepath = "/wrk/users/zosa/codes/pimlico_store/yle_preprocess3/main/lemmatize/lemmas/data/"
	print("Reading lemmatized articles from ", yle_filepath)
	articles = {}
	tar_files = os.listdir(yle_filepath)
	for tar_file in tar_files:
		tar = tarfile.open(yle_filepath + "/" + tar_file, "r")
		for member in tar.getmembers():
			f = tar.extractfile(member)
			if f is not None:
				filename = member.name
				print("Filename:", filename)
				text = f.read().decode('utf-8')
				lines = text.split("|DatePublished ")
				for art in lines:
					if len(art)>0:
						a = art.split("|")
						date_pub = a[0]
						art_no = a[1].split()[1]
						text = a[2]
						lang = "fi" if "fi" in filename else "sv"
						if art_no not in articles.keys():
							articles[art_no] = {}
							d = date_pub.split("-")
							articles[art_no]['date'] = d[0]+d[1]
						articles[art_no][lang] = text
	start_date = date(year=start_year, month=1, day=1)
	start_date_str = str(start_date.year)+"0"+str(start_date.month)
	start_date_int = int(start_date_str)
	if ts == 0:
		end_date = add_months(start_date, n_timeslices-1)
	else:
		end_date = add_months(start_date, ts)
	if end_date.month < 10:
		end_date_str = str(end_date.year)+"0"+str(end_date.month)
	else:
		end_date_str = str(end_date.year)+str(end_date.month)
	end_date_int = int(end_date_str)
	print("Start date int:", start_date_int)
	print("End date int:", end_date_int)
	languages = ['fi', 'sv']
	documents = {lang: [] for lang in languages}
	timestamps = []
	keys = list(articles.keys())
	for k in keys:
		art = articles[k]
		if ts == 0:
			if int(art['date']) <= end_date_int and int(art['date']) >= start_date_int:
				for lang in languages:
					doc = art[lang]
					clean_doc = clean_yle(doc.split()).split()
					documents[lang].append(clean_doc)
				timestamps.append(art['date'])
		else:
			if int(art['date']) == end_date_int and int(art['date']) >= start_date_int:
				for lang in languages:
					doc = art[lang]
					clean_doc = clean_yle(doc.split()).split()
					documents[lang].append(clean_doc)
				timestamps.append(art['date'])
	unique_timestamps = list(set(timestamps))
	unique_timestamps.sort()
	print("Timestamps: ", unique_timestamps)
	documents = {lang: np.array(documents[lang]) for lang in languages}
	timestamps = np.array(timestamps)
	documents_sampled = {lang: [] for lang in languages}
	dictionary = {lang: set() for lang in languages}
	if max_doc_per_timeslice > 0:
		for t in unique_timestamps:
			n_docs = np.sum(timestamps==t)
			print("Docs for timeslice:", n_docs)
			if max_doc_per_timeslice < n_docs:
				random_indexes = random.sample(range(n_docs), max_doc_per_timeslice)
				for index in random_indexes:
					for lang in languages:
						docs_t = documents[lang][timestamps == t]
						documents_sampled[lang].append(docs_t[index])
						dictionary[lang].update(docs_t[index])
	else:
		for lang in languages:
			documents_sampled[lang] = documents[lang]
			for doc in documents[lang]:
				dictionary[lang].update(doc)
	dictionary = {lang: list(dictionary[lang]) for lang in languages}
	print("Documents per language:", len(documents_sampled[languages[0]]))
	return documents_sampled, dictionary




def CalculateCounts(par):
	languages = par['languages']
	for d in range(par['D']):
		for lang in languages:
			for i in range(par['N'][lang][d]):
				topic_di = par['z'][lang][d][i]
				word_di = par['w'][lang][d][i]
				par['m'][lang][d,topic_di] += 1
				par['n'][lang][topic_di, word_di] += 1
				par['n_sum'][lang][topic_di] += 1


def calculate_counts_new_docs(model, lang):
	for d in range(model['D']):
		for i in range(model['N'][lang][d]):
			topic_di = model['z'][lang][d][i]
			model['m'][lang][d, topic_di] += 1
	return model


# m counts: D*K matrix for each language where D = no. of docs, K = no. of topics
# n counts: K*V matrix for each language where K = no. of topics, V = size of vocabulary
def InitializeParameters(documents, dictionary, alpha=1.0, beta=0.1, n_topics=10, n_iter=1000):
	print("Initializing parameters...")
	par = {}
	par['languages'] = list(documents.keys())
	languages = par['languages']
	par['max_iterations'] = n_iter
	par['T'] = n_topics
	par['D'] = len(documents[languages[0]])
	par['V'] = {lang: len(dictionary[lang]) for lang in languages}
	par['N'] = {lang: np.array([len(doc) for doc in documents[lang]]) for lang in languages}
	par['alpha'] = np.array([alpha for _ in range(par['T'])])
	par['beta'] = {lang: np.array([beta for _ in range(par['V'][lang])]) for lang in languages}
	par['beta_sum'] = {lang: sum(par['beta'][lang]) for lang in languages}
	par['word_id'] = {lang: {dictionary[lang][i]: i for i in range(len(dictionary[lang]))} for lang in languages}
	par['word_token'] = {lang: dictionary[lang] for lang in languages}
	par['z'] = {lang: [[random.randrange(0, par['T']) for _ in range(par['N'][lang][d])] for d in range(par['D'])] for lang in languages}
	par['w'] = {lang: [[par['word_id'][lang][documents[lang][d][i]] for i in range(par['N'][lang][d])] for d in range(par['D'])] for lang in languages}
	par['m'] = {lang: np.zeros((par['D'], par['T'])) for lang in languages}
	par['n'] = {lang: np.zeros((par['T'], par['V'][lang])) for lang in languages}
	par['n_sum'] = {lang: np.zeros(par['T']) for lang in languages}
	np.set_printoptions(threshold=np.inf)
	np.seterr(divide='ignore', invalid='ignore')
	CalculateCounts(par)
	for lang in languages:
		print("Vocab size -",lang,":", par['V'][lang])
	return par


def init_updated_model(model, documents, lang, alpha=None):
	print("Initializing parameters...")
	new_model = model
	new_model['D'] = len(documents)
	new_model['N'] = {lang: np.array([len(doc) for doc in documents])}
	new_model['z'] = {lang: [[random.randrange(0, new_model['T']) for _ in range(new_model['N'][lang][d])] for d in range(new_model['D'])]}
	new_model['w'] = {lang: [[new_model['word_id'][lang][documents[d][i]] for i in range(new_model['N'][lang][d])] for d in range(new_model['D'])]}
	new_model['m'] = {lang: np.zeros((new_model['D'], new_model['T']))}
	if alpha is not None:
		print("New alpha:", alpha)
		new_model['alpha'] = np.array([alpha for _ in range(new_model['T'])])
	np.set_printoptions(threshold=np.inf)
	np.seterr(divide='ignore', invalid='ignore')
	new_model = calculate_counts_new_docs(new_model, lang)
	return new_model


def compute_phi_post(par):
	# compute phi
	languages = par['languages']
	phi = {}
	beta_mat = {}
	for lang in languages:
		beta_mat_lang = np.tile(par['beta'][lang], (par['T'],1))
		beta_mat[lang] = beta_mat_lang
	for lang in languages:
		phi_lang = par['n'][lang] + beta_mat[lang]
		phi_lang = phi_lang/phi_lang.sum(axis=1)[:, None]
		phi[lang] = phi_lang
	return phi


def compute_theta_post(par, lang=None):
	# compute theta
	alpha_mat = np.tile(par['alpha'], (par['D'],1))
	ones_m_counts = np.ones((par['D'], par['T']))
	if lang is None:
		theta = ones_m_counts * (par['m']['fi'] + par['m']['sv']) + alpha_mat
	else:
		theta = ones_m_counts * (par['m'][lang]) + alpha_mat
	theta = theta/theta.sum(axis=1)[:, None]
	return theta


def infer_new_docs(new_model, lang, max_iter=200):
	print("Inferring document vectors for", new_model['D'], "new docs...")
	alpha_mat = np.tile(new_model['alpha'], (new_model['D'], 1))
	beta_mat_lang = np.tile(new_model['beta'][lang], (new_model['T'], 1))
	ones_m_counts = np.ones((new_model['D'], new_model['T']))
	theta_doc = ones_m_counts * new_model['m'][lang] + alpha_mat
	theta_intermediate = []
	for iteration in range(max_iter):
		#print("Iteration", iteration,"of",max_iter)
		for d in range(new_model['D']):
			n_sum = new_model['n'][lang] + beta_mat_lang
			denominator = new_model['n_sum'][lang] + new_model['beta_sum'][lang]
			for i in range(new_model['N'][lang][d]):
				#print("Word",i)
				word_di = new_model['w'][lang][d][i]
				old_topic = new_model['z'][lang][d][i]
				new_model['m'][lang][d, old_topic] -= 1.0
				#new_model['n'][lang][old_topic, word_di] -= 1.0
				#new_model['n_sum'][lang][old_topic] -= 1.0
				# vectorized computation of topic probabilities
				phi_lang = n_sum[:,word_di] / denominator
				topic_probabilities = theta_doc[d] * phi_lang
				sum_topic_probabilities = np.sum(topic_probabilities)
				if sum_topic_probabilities == 0:
					topic_probabilities = np.full((new_model['T'],), 1.0/new_model['T'])
				else:
					#topic_probabilities = topic_probabilities / sum_topic_probabilities
					topic_probabilities = np.asarray(topic_probabilities).astype('float64')
					topic_probabilities /= topic_probabilities.sum()
				#topic_probabilities = topic_probabilities.astype('float64')
				#print("Topic prob: ", topic_probabilities)
				#print("Sum topic prob:", topic_probabilities.sum())
				#res = np.random.multinomial(1, topic_probabilities, size=1)
				#print("Res: ", res)
				new_topic = list(np.random.multinomial(1, topic_probabilities, size=1)[0]).index(1)
				new_model['z'][lang][d][i] = new_topic
				new_model['m'][lang][d, new_topic] += 1
		if (iteration+1) % 25 == 0:
			theta_25 = compute_theta_post(new_model, lang=lang)
			theta_intermediate.append(theta_25)
	#theta_new_docs = compute_theta_post(new_model, lang=lang)
	theta_new_docs = np.mean([theta for theta in theta_intermediate], axis=0)
	print("Done!")
	return theta_new_docs


def get_document_vectors(model_file, articles, lang, alpha=None):
	#lang = "fi"
	#model_file = "trained_models/yle_100topics"
	model = pickle.load(open(model_file+".pkl",'rb'))
	print("----- Model parameters for", lang.upper(), "-----")
	print("topics =", model['T'])
	print("vocab per lang =", model['V']['fi'])
	print("docs per lang =", model['D'])
	print("iterations =", model['max_iterations'])
	print("alpha =", model['alpha'][0])
	print("beta =", model['beta']['fi'][0])
	#path = "../data/yle"
	#articles_dict, subjects = process_articles(path)
	#articles = [articles_dict['fi'][i]['content'] for i in range(5)]
	valid_words = model['word_token'][lang]
	articles = [clean_yle_new_articles(art.split(), valid_words) for art in articles]
	new_model = init_updated_model(model, articles, lang)
	doc_vectors = infer_new_docs(new_model, lang)
	return doc_vectors


def BildaGibbsSampling(par):
	languages = par['languages']
	print("Starting MLTM training for", languages)
	alpha_mat = np.tile(par['alpha'], (par['D'],1))
	beta_mat = {}
	for lang in languages:
		beta_mat_lang = np.tile(par['beta'][lang], (par['T'],1))
		beta_mat[lang] = beta_mat_lang
	ones_m_counts = np.ones((par['D'], par['T']))
	for iteration in range(par['max_iterations']):
		print("Iteration", str(iteration+1), "of", par['max_iterations'])
		for d in range(par['D']):
			theta_doc = ones_m_counts * (par['m']['fi'] + par['m']['sv']) + alpha_mat
			for lang in languages:
				n_sum = par['n'][lang] + beta_mat[lang]
				denominator = par['n_sum'][lang] + par['beta_sum'][lang]
				for i in range(par['N'][lang][d]):
					word_di = par['w'][lang][d][i]
					old_topic = par['z'][lang][d][i]
					par['m'][lang][d,old_topic] -= 1.0
					par['n'][lang][old_topic,word_di] -= 1.0
					par['n_sum'][lang][old_topic] -= 1.0
					# vectorized computation of topic probabilities
					phi_lang = n_sum[:,word_di] / denominator
					topic_probabilities = theta_doc[d] * phi_lang
					sum_topic_probabilities = np.sum(topic_probabilities)
					if sum_topic_probabilities == 0:
						topic_probabilities = np.full((par['T'],), 1.0/par['T'])
					else:
						topic_probabilities = topic_probabilities / sum_topic_probabilities
					#print("Topic prob: ", topic_probabilities)
					new_topic = list(np.random.multinomial(1, topic_probabilities, size=1)[0]).index(1)
					par['z'][lang][d][i] = new_topic
					par['m'][lang][d,new_topic] += 1
					par['n'][lang][new_topic,word_di] += 1
					par['n_sum'][lang][new_topic] += 1
	par['theta'] = compute_theta_post(par)
	par['phi'] = compute_phi_post(par)
	print("Finished training MLTM!")
	return par