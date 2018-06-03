import artm
import pandas as pd
import numpy as np
import sys
import pandas as pd
import gensim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import collections
import operator
import json
import gensim


def make_file(lines, ngram_range, LOGS_DATA_PATH):
    if ngram_range == (1, 1):
        with open(LOGS_DATA_PATH, 'w') as wf:
            for index in range(len(lines)):
                wf.write("doc{0} {1}\n".format(index, lines[index]))
    
    if ngram_range == (2, 2):
        with open(LOGS_DATA_PATH, 'w') as wf:
            for index in range(len(lines)):
                tokens = [x for x in lines[index].strip().split(" ") if x != ""]
                s = ""
                for i in range(len(tokens) -1):
                    s += "{0}_{1} ".format(tokens[i],  tokens[i+1])
                wf.write("doc{0} {1}\n".format(index, s))

    if ngram_range == (1, 2):
        with open(LOGS_DATA_PATH, 'w') as wf:
            for index in range(len(lines)):
                tokens = [x for x in lines[index].strip().split(" ") if x != ""]
                s = ""
                for i in range(len(tokens) -1):
                    s += "{0}_{1} {0} ".format(tokens[i],  tokens[i+1])
                s += tokens[-1]
                wf.write("doc{0} {1}\n".format(index, s))



def pipeline_plsa_bigartm(lines, TOPIC_NUMBER,  ngram_range, topnwords, LOGS_DATA_PATH="plsa.txt", TARGET_FOLDER="plsa"):

    make_file(lines, ngram_range, LOGS_DATA_PATH)

    batch_vectorizer = artm.BatchVectorizer(data_path=LOGS_DATA_PATH, 
                                            data_format='vowpal_wabbit',  
                                            target_folder=TARGET_FOLDER)
    
    model_artm = artm.ARTM(num_topics=TOPIC_NUMBER, cache_theta=True)
    model_artm.initialize(dictionary=batch_vectorizer.dictionary)
    
    model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=0.05))
    model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=1.5e+5))
    model_artm.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta',tau=-0.01))

    model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
    model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
    model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=topnwords), overwrite=True)
    model_artm.scores.add(artm.PerplexityScore(name='PerplexityScore',
                                                        dictionary=batch_vectorizer.dictionary))

    model_artm.num_document_passes = 2
    model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)
    
    topic_names = {}
    for topic_name in model_artm.topic_names:
        topic_names[topic_name] = model_artm.score_tracker['TopTokensScore'].last_tokens[topic_name]
    
    #return label_after_bigarm(model_artm),  topic_names
    return "nothing, sorry",  topic_names


def label_after_bigarm(model):
    theta_matrix = model.get_theta()
    theta_matrix = theta_matrix.transpose()

    topic_distribution = {}
    labels = []
    print(theta_matrix.size)
    print(theta_matrix.index)
    for i in theta_matrix.index:
        th = np.argmax(theta_matrix.iloc[i])
        labels.append(th)

    return labels


def lsi(lines, n_clusters, top_n_words, ngram_range):
    if ngram_range == (1, 1):
        texts = [[word for word in line.lower().split(" ") if word != ''] for line in lines]
    else:
        texts = []
        for line in lines:
            local = []
            tokens = line.split(" ")
            for i in range(len(tokens)-1):
                if tokens[i] != '' and tokens[i+1] != '':
                    local.append("{0} {1}".format(tokens[i], tokens[i+1]))
                    if ngram_range == (1, 2):
                        local.append(tokens[i])
            local.append(tokens[-1])
            texts.append(local)   
        
    dictionary = gensim.corpora.Dictionary(texts)
    corp = [dictionary.doc2bow(text) for text in texts]
    tfidf = gensim.models.TfidfModel(corp) 
    corpus_tfidf = tfidf[corp]
    lsi = gensim.models.LsiModel(corpus=corp, id2word=dictionary, num_topics=n_clusters)

    topic_dict = {}
    for i in range(n_clusters):
        tpc = lsi.show_topic(i, top_n_words)
        topic = [t[0] for t in tpc]
        topic_dict[i] = topic
        
    return lsi, topic_dict  


def pipeline_lda_bigartm(lines, n_clusters,  ngram_range, topnwords, LOGS_DATA_PATH="plsa.txt", TARGET_FOLDER="plsa"):
    
    make_file(lines, ngram_range, LOGS_DATA_PATH)

    bv = artm.BatchVectorizer(data_path=LOGS_DATA_PATH, 
                                            data_format='vowpal_wabbit',  
                                            target_folder=TARGET_FOLDER)
    
    lda = artm.LDA(num_topics=n_clusters, alpha=0.01, beta=0.001, cache_theta=True,
                dictionary=bv.dictionary)
    lda.fit_offline(batch_vectorizer=bv)

    top_tokens = lda.get_top_tokens(num_tokens=topnwords)
    topic_names = {}
    for i, token_list in enumerate(top_tokens):
        topic_names[i] = token_list

    return label_after_bigarm(lda),  topic_names 



def kmeans_tfidf(ngram_range, min_df, lines, n_clusters, num):
    vectorizer1 = TfidfVectorizer(ngram_range= ngram_range, min_df=min_df)
    tfidf_matrix1 = vectorizer1.fit_transform(lines)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=17)
    kmeans.fit(tfidf_matrix1)
    
    filename = "kmeans_tfidf_{0}{1}_{2}_{3}.json".format(ngram_range[0], ngram_range[1], n_clusters, num)
    return kmeans.labels_, get_top_words_tfidf(kmeans, vectorizer1, num, filename)


def get_top_words_tfidf(kmeans, vectorizer,  num, filename):
    top_n_words = {}
    for clust in range(kmeans.n_clusters):
        v = kmeans.cluster_centers_[clust]
        superdict = {}
        for i in range(len(v)):
            if v[i] > 0:
                superdict[vectorizer.get_feature_names()[i]] = v[i]
        supsort = sorted(superdict.items(), key=operator.itemgetter(1), reverse=True)
        nwords = [x[0] for x in supsort[:num]]
        top_n_words[clust] = nwords
    
    with open(filename, "w") as f:
        f.write(json.dumps(top_n_words, ensure_ascii=False))
    return top_n_words



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF

def get_nmf_topics(model, n_top_words, vectorizer, n_clusters):
    feat_names = vectorizer.get_feature_names()
    word_dict = {};
    for i in range(n_clusters):
        words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict[i+1] = words;
    return  word_dict

def nnmf(lines, n_clusters, ngram_range, topnwords):
    transformer = TfidfVectorizer(ngram_range= ngram_range, min_df=5)
    x_tfidf = transformer.fit_transform(lines)
    xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
    model = NMF(n_components=n_clusters, init='nndsvd');
    model.fit(x_tfidf)
    topics = get_nmf_topics(model, topnwords, transformer, n_clusters)
    return model, topics



import nltk
from nltk import FreqDist
from nltk.util import ngrams
def count_corpus_frequency2(sentences):
    sentences_tokens = [line.strip().split(" ") for line in sentences]
    
    freq1 = FreqDist()
    freq12 = FreqDist() 
    freq2 = FreqDist()
    
    for sentence in sentences_tokens:
        sentence = [x for x in sentence if x!='']
        bigrams = ["{0} {1}".format(t[0], t[1]) for t in ngrams(sentence, 2)]

        freq1.update(sentence)
        freq12.update(sentence)
        freq12.update(bigrams)
        freq2.update(bigrams)
    
    return freq1, freq12, freq2

def count_pair_frequency2(w1, w2, lines):
        count = 0
        for line in lines:
            if w1 in line and w2 in line:
                count += 1
        return count

def count_pmi_npmi2(w1, w2, lines, general_freq):
    N = len(lines)
    px = (1+general_freq[w1])
    py = (1+general_freq[w2])
    pxy = (1+count_pair_frequency(w1, w2, lines))
    pmi = np.log(N*pxy/(px*py))
    npmi = pmi/(-np.log(pxy/N))
    return pmi, npmi

def count_pmi_npmi(x, y, lines):
    count_x = 1
    count_xy = 1
    count_y = 1
    for line in lines:
        if x in line and y in line:
            count_xy += 1
        if x in line:
            count_x += 1
        if y in line:
            count_y += 1
    pmi = np.log(len(lines)*count_xy/(count_x*count_y))
    npmi = pmi/(-np.log(count_xy/len(lines)))
    return pmi, npmi


def prepare_topwords(old):
    n = {}
    for k in old:
        r = [x.replace("_", " ") for x in old[k]]
        n[k] = r
    return n


def get_metrics(topnwords, lines, topwords_number):
    pmi_by_topic = []
    npmi_by_topic = []
    for key in topnwords:
        if len(topnwords[key]) < topwords_number:
            pmi_by_topic.append(0)
            npmi_by_topic.append(0)
        else:
            npmi_scores = []
            pmi_scores = []
            for w1 in topnwords[key]:
                for w2 in topnwords[key]:
                    if w1 != w2:
                        pmi, npmi = count_pmi_npmi(w1, w2, lines)
                        pmi_scores.append(pmi)
                        npmi_scores.append(npmi)
            pmi_by_topic.append(np.array(pmi_scores).mean())
            npmi_by_topic.append(np.array(npmi_scores).mean())
    return np.array(pmi_by_topic), np.array(npmi_by_topic)


def pipeline_lda_gensim(lines, numclusters, ngram_range, num_nwords):
    if ngram_range == (1, 1):
        sentences_tokens = [line.strip().split(" ") for line in lines]
    if ngram_range == (1, 2):
        sentences_tokens = []
        for index in range(len(lines)):
            tokens = [x for x in lines[index].strip().split(" ") if x != ""]
            s = []
            for i in range(len(tokens) -1):
                s.append(tokens[i])
                s.append("{0}_{1}".format(tokens[i],  tokens[i+1]))
            s.append(tokens[-1])
            sentences_tokens.append(s)
    if ngram_range == (2, 2):
        sentences_tokens = []
        for index in range(len(lines)):
            tokens = [x for x in lines[index].strip().split(" ") if x != ""]
            s = []
            for i in range(len(tokens) -1):
                s.append("{0}_{1}".format(tokens[i],  tokens[i+1]))
            sentences_tokens.append(s)
        
    dictt = gensim.corpora.Dictionary(sentences_tokens)
    corpus = [dictt.doc2bow(text) for text in sentences_tokens]
    
    ldamodel = gensim.models.LdaModel(corpus, num_topics=numclusters, alpha='auto', passes=2)
    
    topnwords = {}
    for topic in range(ldamodel.num_topics):
        topnwords[topic]= [dictt[x[0]] for x in ldamodel.get_topic_terms(topic, topn= num_nwords)]
    return topnwords


def pipeline_word2vec(lines, n_clusters, topn_words):
    sentences = [line.split(" ") for line in lines]
    model = gensim.models.Word2Vec(sentences, size=100, window=3, min_count=5, workers=4)
    matrix = w2v_matrix(lines, model)
    kmeans = KMeans(n_clusters=20, random_state=17)
    kmeans.fit(matrix)
    
    topwords = {}
    for i in range(kmeans.n_clusters):
        x = model.similar_by_vector(kmeans.cluster_centers_[i], topn=20)
        topwords[i] = [y[0] for y in x]

    return topwords


def w2v_matrix(lines, model):
    matrix = np.zeros(shape=(1,100))
    for sent in lines:
        matrix = np.vstack([matrix, get_average_vector(model, sent)])
    return matrix
    
    
def get_average_vector(model, sent):
    vectors = []
    for word in sent.split(" "):
        if word in model.wv.vocab:
            vectors.append(model.wv[word])
    if len(vectors)> 1:
        return np.array(vectors).mean(axis=0)
    return np.zeros((1, 100))