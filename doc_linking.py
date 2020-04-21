from mltm import get_document_vectors
from sklearn.metrics.pairwise import cosine_similarity
from wasserstein import load_embeddings, clean_corpus_using_embeddings_vocabulary, WassersteinDistances
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from gensim.models import KeyedVectors
from scipy.stats import entropy
import tarfile
import os
import numpy as np
from nltk.corpus import stopwords
import string
import pickle

stopwords_yle = set(stopwords.words('finnish')).union(stopwords.words('swedish'))
exclude = set(string.punctuation)


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def compute_mrr(pred, actual):
    mrr_all = []
    for i in range(len(pred)):
        pred_doc = pred[i]
        actual_doc = actual[i]
        ranks_doc = [pred_doc.index(a) for a in actual_doc if a in pred_doc]
        first_rank = min(ranks_doc)
        mrr_doc = 1/(first_rank+1)
        mrr_all.append(mrr_doc)
    mrr_all = np.mean(mrr_all)
    return mrr_all


def compute_precision_at_n(pred_ranking, true_related, n):
    prec_all = []
    for i, pred_art in enumerate(pred_ranking):
        pred_art = pred_art[:n]
        true_art = true_related[i]
        tp_art = len(set(true_art).intersection(set(pred_art)))
        prec_art = tp_art/len(pred_art)
        prec_all.append(prec_art)
    prec_mean = np.mean(prec_all)
    return prec_mean


def clean_article_yle(doc):
    clean_short = " ".join([tok for tok in doc if len(tok) > 2 and tok not in stopwords_yle])
    clean_punc = ''.join(ch for ch in clean_short if ch not in exclude).lower()
    return clean_punc


#Cr5 embeddings are downloaded from the Cr5 repo (Josifoski et al., 2017)
def load_cr5_embeddings():
    cr5_embeddings = {}
    languages = ['fi', 'sv']
    cr5_filepath = "../trained_models/cr5_embeddings/"
    for lang in languages:
        cr5_file = cr5_filepath+lang+"_2013_2015_monthly_cr5.txt"
        cr5_model = KeyedVectors.load_word2vec_format(fname=cr5_file, binary=False)
        cr5_embeddings[lang] = cr5_model
    return cr5_embeddings


def get_cr5_similar_docs_and_distance(fi_doc_ids, sv_doc_ids, sv_keys_dict, query_lang='fi', target_lang='sv'):
    print("\nGetting Cr5 predictions...")
    cr5_embeddings = load_cr5_embeddings()
    sv_num_docs = len(cr5_embeddings[target_lang].vocab)
    dist_mat = np.zeros((len(fi_doc_ids), len(sv_doc_ids)))
    sv_doc_names = list(cr5_embeddings[target_lang].vocab)
    fi_doc_names = list(cr5_embeddings[query_lang].vocab)
    random_vec = np.random.rand(1,300)
    most_similar_list = []
    for i,doc_id in enumerate(fi_doc_ids):
        fi_doc_name = 'doc' + str(doc_id)
        if fi_doc_name in fi_doc_names:
            query_doc_vec = cr5_embeddings[query_lang][fi_doc_name]
        else:
            query_doc_vec = cr5_embeddings[query_lang][fi_doc_names[0]]
        most_similar_docs = cr5_embeddings[target_lang].most_similar(positive=[query_doc_vec],topn=sv_num_docs)
        most_similar_ids = [d[0][3:] for d in most_similar_docs]
        most_similar_ids_valid = [sim_id for sim_id in most_similar_ids if sim_id in sv_keys_dict]
        most_similar_list.append(most_similar_ids_valid)
        for j,sv_doc_id in enumerate(sv_doc_ids):
            sv_doc_name = 'doc'+str(sv_doc_id)
            if sv_doc_name in sv_doc_names:
                # document embedding for an SV news article
                sv_doc_vec = cr5_embeddings[target_lang][sv_doc_name]
                cosine_dist = 1-(abs(cosine_similarity([query_doc_vec], [sv_doc_vec])))
                #cosine_dist = cosine([query_doc_vec], [sv_doc_vec])
                dist_mat[i, j] = cosine_dist
    return dist_mat, most_similar_list


# Wasserstein script is downloaded from the Wasserstein repo (Balikas, et al. 2018)
def get_wasserstein_similar_docs_and_distance(fi_articles, sv_articles, chunk_size=50, query_lang='fi', target_lang='sv', entropic_reg=0.1, use_cr5_emb=False):
    print("\nGetting Wasserstein predictions...")
    wasserstein_pred = []
    fi_emb = "../cross_lingual_embeddings/concept_net_1706.300.fi"
    sv_emb = "../cross_lingual_embeddings/concept_net_1706.300.sv"
    if use_cr5_emb==True:
        fi_emb = "../cross_lingual_embeddings/joint_28_fi.txt"
        sv_emb = "../cross_lingual_embeddings/joint_28_sv.txt"
    vectors_fi = load_embeddings(fi_emb, 300)
    vectors_sv = load_embeddings(sv_emb, 300)
    fi = [clean_article_yle(f.split()) for f in fi_articles]
    sv = [clean_article_yle(s.split()) for s in sv_articles]
    #stopwords_all = set(nltk.corpus.stopwords.words("finnish")).update(set(nltk.corpus.stopwords.words("swedish")))
    clean_fi, clean_vectors_fi, keys_fi = clean_corpus_using_embeddings_vocabulary(set(vectors_fi.keys()), fi, vectors_fi, query_lang, stopwords_yle)
    clean_sv, clean_vectors_sv, keys_sv = clean_corpus_using_embeddings_vocabulary(set(vectors_sv.keys()), sv, vectors_sv, target_lang, stopwords_yle)
    num_target_docs = len(clean_sv)
    num_query = len(clean_fi)
    # divide the query docs into chunks of 50 so the distance computation is not infeasible
    num_chunks = int(num_query/chunk_size)+1
    start_indexes = list(np.linspace(start=0, stop=num_query, num=num_chunks, dtype=int))
    print("Chunk indexes:", start_indexes)
    distances = []
    predictions = []
    for i in range(len(start_indexes)-1):
        start_chunk = start_indexes[i]
        end_chunk = start_indexes[i+1]
        clean_fi_chunk = list(clean_fi[start_chunk:end_chunk])
        print("Chunk size:", len(clean_fi_chunk))
        clean_sv = list(clean_sv)
        our_corpus = clean_fi_chunk + clean_sv
        vocab = CountVectorizer().fit(our_corpus)  # get the vocabulary of the corpus
        common = [word for word in vocab.get_feature_names() if word in clean_vectors_fi or word in clean_vectors_sv]
        W_common = []
        for w in common:  # Similarly, to save memory keep only the embeddings of words that appear in the corpus.
            if w in clean_vectors_fi:
                W_common.append(np.array(clean_vectors_fi[w]))
            else:
                W_common.append(np.array(clean_vectors_sv[w]))
        print("The vocabulary size is:", len(W_common))
        W_common = np.array(W_common)
        W_common = normalize(W_common)
        vect = TfidfVectorizer(vocabulary=common, dtype=np.double, norm=None, )
        vect.fit(clean_fi_chunk + clean_sv)
        X_query_idf = vect.transform(clean_fi_chunk)  # tf-idf representation of query doc
        X_target_idf = vect.transform(clean_sv)  # tf-idf representation of the target documents
        print("X_query_idf shape:", X_query_idf.shape)
        print("X_target_idf shape:", X_target_idf.shape)
        clf = WassersteinDistances(W_embed=W_common, n_neighbors=5, n_jobs=10, sinkhorn_reg=entropic_reg)
        clf.fit(X_target_idf[:num_target_docs], np.ones(num_target_docs))
        dist_chunk, pred_chunk = clf.kneighbors(X_query_idf[:num_target_docs], n_neighbors=num_target_docs)
        distances.append(dist_chunk)
        predictions.append(pred_chunk)
    distances = np.stack(distances, axis=0)
    distances = np.concatenate((distances), axis=0)
    predictions = np.concatenate(predictions, axis=0)
    print("distances shape:", distances.shape)
    print("predictions shape:", predictions.shape)
    return distances, predictions


def get_topic_model_similar_docs_and_distance(model_name, fi_articles, sv_articles, query_lang='fi', target_lang='sv', alpha=None):
    print("\nGet prediction from topic models...")
    fi_doc_vecs = get_document_vectors(model_name, fi_articles, lang=query_lang, alpha=alpha)
    sv_doc_vecs = get_document_vectors(model_name, sv_articles, lang=target_lang, alpha=alpha)
    pickle_file = open('sv_doc_vecs_' + model_name + '.pkl', 'wb')
    pickle.dump(sv_doc_vecs, pickle_file)
    pickle_file.close()
    # Compute document distances using JS-Divergence
    dist_mat = np.zeros((fi_doc_vecs.shape[0], sv_doc_vecs.shape[0]))
    for i in range(len(fi_doc_vecs)):
        for j in range(len(sv_doc_vecs)):
            jsd = compute_jsd(fi_doc_vecs[i], sv_doc_vecs[j])
            dist_mat[i, j] = jsd
    pred_ranking = []
    for i in range(len(fi_doc_vecs)):
        sorted_rank = [i for _, i in sorted(zip(dist_mat[i, :], range(len(dist_mat[i, :]))), reverse=False)]
        pred_ranking.append(sorted_rank)
    return dist_mat, pred_ranking


def get_topic_model_predictions(tm_dist):
    pred_ranking = []
    num_query = len(tm_dist)
    for i in range(num_query):
        sorted_rank = [i for _, i in sorted(zip(tm_dist[i, :], range(len(tm_dist[i, :]))), reverse=False)]
        pred_ranking.append(sorted_rank)
    return pred_ranking


def ensemble_ranker(ranking1, ranking2, ensemble_param=0.5):
    ensemble_ranking = []
    for i in range(len(ranking1)):
        doc_scores = []
        doc_rank1 = ranking1[i]
        doc_rank2 = ranking2[i]
        unranked_val = len(doc_rank1) + 1
        for j,d in enumerate(doc_rank1):
            rank1 = j+1
            if d in doc_rank2:
                rank2 = doc_rank2.index(d)+1
            else:
                rank2 = unranked_val
            score = (ensemble_param*rank1) + ((1-ensemble_param)*rank2)
            doc_scores.append(score)
        sorted_rank = [rank for _, rank in sorted(zip(doc_scores, doc_rank1), reverse=False)]
        ensemble_ranking.append(sorted_rank)
    return ensemble_ranking


def ensemble_ranker2(ranking1, ranking2, ranking3):
    ensemble_ranking = []
    for i in range(len(ranking1)):
        doc_scores = []
        doc_rank1 = ranking1[i]
        doc_rank2 = ranking2[i]
        doc_rank3 = ranking3[i]
        unranked_val = len(doc_rank1) + 1
        for j,d in enumerate(doc_rank1):
            rank1 = j+1
            if d in doc_rank2:
                rank2 = doc_rank2.index(d)+1
            else:
                rank2 = unranked_val
            if d in doc_rank3:
                rank3 = doc_rank3.index(d)+1
            else:
                rank3 = unranked_val
            score = (rank1+rank2+rank3)/3
            doc_scores.append(score)
        sorted_rank = [rank for _, rank in sorted(zip(doc_scores, doc_rank1), reverse=False)]
        ensemble_ranking.append(sorted_rank)
    return ensemble_ranking


# all distances are in numpy.matrix where n_rows = num of query docs, n_cols = num of target docs
def ensemble_dist_ranker(dist1, dist2, dist3=None, ensemble_method='mean'):
    num_docs = dist1.shape[0]
    ensemble_doc_rank = []
    for i in range(num_docs):
        if dist3 is None:
            dist = np.array([dist1[i],dist2[i]])
        else:
            dist = np.array([dist1[i],dist2[i],dist3[i]])
        if ensemble_method == 'mean':
            dist = list(np.mean(dist, axis=0))
        else:
            dist = list(np.max(dist, axis=0))
        new_doc_rank = [rank for _, rank in sorted(zip(dist, range(len(dist))), reverse=False)]
        ensemble_doc_rank.append(new_doc_rank)
    return ensemble_doc_rank


# all distances are in numpy.matrix where n_rows = num of query docs, n_cols = num of target docs
def ensemble_dist_ranker_weighted(dist1, dist2, param=0.5):
    num_docs = dist1.shape[0]
    ensemble_doc_rank = []
    for i in range(num_docs):
        dist = np.array([param*dist1[i], (1 - param)*dist2[i]])
        dist = list(np.mean(dist, axis=0))
        new_doc_rank = [rank for _, rank in sorted(zip(dist, range(len(dist))), reverse=False)]
        ensemble_doc_rank.append(new_doc_rank)
    return ensemble_doc_rank


# normalize Wasserstein distances
def get_normalized_distance(dist):
    row_sums = dist.sum(axis=1)
    normalized = dist / row_sums[:, np.newaxis]
    return normalized


def get_mean_centered_distance(dist):
    row_means = dist.mean(axis=1)
    normalized = dist - row_means[:, np.newaxis]
    return normalized


def get_lemmatized_docs():
    yle_filepath = "filepath to Yle lemmatised corpus"
    print("Reading articles from ", yle_filepath)
    languages = ['fi', 'sv']
    articles = {lang: {} for lang in languages}
    tar_files = os.listdir(yle_filepath)
    for tar_file in tar_files:
        tar = tarfile.open(yle_filepath + "/" + tar_file, "r")
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                filename = member.name
                print("Filename:", filename)
                if 'fi' in filename:
                    lang = 'fi'
                else:
                    lang = 'sv'
                text = f.read().decode('utf-8')
                article_text = text.split("Article_id : ")
                for art in article_text:
                    if len(art)>10:
                        art_split = art.split("||")
                        art_id = art_split[0]
                        art_content = art_split[1]
                        articles[lang][art_id] = art_content
    return articles


def get_test_dataset(aligned_data, num_articles=1000, year=2013, month=1, query_lang='fi'):
    fi_keys = list(aligned_data.keys())
    fi_keys_month = []
    for key in fi_keys:
        art_date = aligned_data[key]['date'].split("-")
        art_year = int(art_date[0])
        art_month = int(art_date[1])
        if art_year == year and art_month == month:
            fi_keys_month.append(key)
    start_index = 0
    if query_lang == 'sv':
        if (year == 2013 or year == 2015) and month == 1:
            start_index = 100
    elif query_lang == 'fi':
        if ((month == 4 or month == 7) and year != 2015) or (month == 5 and year == 2014):
          start_index = 100
        elif (month == 4 or month == 7) and year == 2015:
          start_index = 300
    fi_keys_valid_double = fi_keys_month[start_index:start_index+(num_articles*2)]
    fi_keys_valid = fi_keys_month[start_index:start_index+num_articles]
    fi_articles = [aligned_data[key]['content'] for key in fi_keys_valid]
    sv_articles_dict = {}
    actual_related = []
    for key in fi_keys_valid_double:
        art = aligned_data[key]
        if 'related_articles' in art:
            related_list = []
            related_articles = art['related_articles']
            # print("Related:", len(related_articles))
            for related in related_articles:
                sv_art_id = related['article_id']
                # print("Article id:", sv_art_id)
                if sv_art_id not in sv_articles_dict:
                    sv_articles_dict[sv_art_id] = related['content']
                related_list.append(sv_art_id)
            actual_related.append(related_list)
    sv_articles = [sv_articles_dict[key] for key in sv_articles_dict]
    sv_keys_dict = dict(zip(list(sv_articles_dict.keys()), range(len(sv_articles))))
    sv_keys = list(sv_keys_dict.keys())
    for i in range(len(actual_related)):
        for j in range(len(actual_related[i])):
            actual_related[i][j] = sv_keys_dict[actual_related[i][j]]
    return fi_articles, sv_articles, fi_keys_valid, sv_keys, sv_keys_dict, actual_related


def save_distance_results(dist_mat, predictions, fi_keys, sv_keys, sv_keys_dict, ground_truth, out_filepath):
    results_dict = {"sv_keys": sv_keys, "fi_keys": fi_keys, "distances": dist_mat, "predictions": predictions, "sv_keys_dict": sv_keys_dict, "ground_truth": ground_truth}
    f = open(out_filepath + ".pkl", "wb")
    pickle.dump(results_dict, f)
    f.close()


