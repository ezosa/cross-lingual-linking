import os
import json
import pickle
from datetime import date

languages = ['fi', 'sv']


def process_articles(path="../data/yle", start_year=2012, end_year=2014):
    articles = {lang: [] for lang in languages}
    None_id = '18-3626'
    for lang in languages:
        print("Processing articles in: ", lang)
        path_lang = path+"/"+lang
        years = os.listdir(path_lang)
        years.sort()
        selected_years = range(start_year, end_year+1)
        selected_years = [str(y) for y in selected_years]
        for y in selected_years:
            path_year = path_lang+"/"+y
            months = os.listdir(path_year)
            months.sort()
            for m in months:
                path_month = path_year+"/"+m
                files = os.listdir(path_month)
                files.sort(reverse=True)
                for f in files:
                    path_file = path_month+"/"+f
                    json_file = open(path_file,'r')
                    json_dict = json.load(json_file)
                    data = json_dict['data']
                    print("Processing file:", path_file)
                    for art in data:
                        article_id = art['id']
                        date_pub = art['datePublished']
                        date_pub = date_pub.split("-")
                        year_pub = date_pub[0]
                        month_pub = date_pub[1]
                        day_pub = date_pub[-1].split("T")[0]
                        date_formatted = date(year=int(year_pub), month=int(month_pub), day=int(day_pub))
                        headline = art['headline']['full']
                        art_content = art['content']
                        content = ""
                        for con in art_content:
                            if 'type' in con.keys():
                                if con['type'] == 'text':
                                    content += con['text'] + " "
                        subject_list = []
                        subject_id_list = []
                        if "subjects" in art.keys():
                            subjects = art['subjects']
                            for sub in subjects:
                                subj = {}
                                subj['title'] = sub['title']['fi']
                                subj['id'] = sub['id']
                                if subj['id'] != None_id:
                                    subject_list.append(subj['title'])
                                    subject_id_list.append((subj['id']))
                                    # if subj['id'] not in subject_dict.keys():
                                    #     subject_dict[subj['id']] = {}
                                    #     subject_dict[subj['id']]['title'] = subj['title']
                                    #     subject_dict[subj['id']]['count'] = 0
                                    # subject_dict[subj['id']]['count'] += 1
                        a = {"date": date_formatted, "headline": headline, "content": content, "article_id": article_id}
                        if len(subject_list) > 0 and len(subject_id_list) > 0:
                            a['subjects'] = subject_list
                            a['subject_ids'] = subject_id_list
                            articles[lang].append(a)
    return articles


def align_articles_one_to_one(articles):
    #align articles using date and named entities
    aa = {lang: [] for lang in languages}
    unmatched = {lang: [] for lang in languages}
    aa_count = 0
    un_count = 0
    for art_fi in articles['fi']:
        date_fi = art_fi['date']
        if date_fi.year:
            if 'subjects' in art_fi.keys():
                subjects_fi = [s for s in art_fi['subjects'] if s is not None]
                for art_sv in articles['sv']:
                    day_delta = (art_sv['date'] - date_fi).days
                    if abs(day_delta) <= 3: #check Swedish articles published 2 days before/after the Finnish article
                        #extract relevant NE from the Swedish article
                        text_sv = art_sv['content']
                        subjects_sv = [s for s in subjects_fi if s in text_sv]
                        #check if the articles share 3 or more NEs
                        inter = list(set(subjects_fi).intersection(set(subjects_sv)))
                        if len(subjects_sv) >= 7:
                            aa['fi'].append(art_fi)
                            aa['sv'].append(art_sv)
                            aa_count += 1
                            print(date_fi)
                            print("Aligned articles:", aa_count)
                            #articles['sv'].remove(art_sv)
                            break
                        # store unmatched articles for validation/testing
                        else:
                            unmatched['fi'].append(art_fi)
                            unmatched['sv'].append(art_sv)
                            un_count += 1
                            #print("Unmatched articles: ", un_count)
                    elif day_delta >= 30:
                        break
    print("Total aligned articles: ", aa_count)
    print("Total unmatched articles: ", un_count)
    return aa, unmatched


# link one Finnish news article to one or more Swedish articles
def align_articles_one_to_many(articles):
    aligned_articles = {}
    for art_fi in articles['fi']:
        print("Aligning Finnish article", art_fi['article_id'])
        date_fi = art_fi['date']
        if date_fi.year:
            if 'subjects' in art_fi.keys():
                subjects_fi = [s for s in art_fi['subjects'] if s is not None]
                for art_sv in articles['sv']:
                    day_delta = (art_sv['date'] - date_fi).days
                    if abs(day_delta) <= 30: #check Swedish articles published 2 days before/after the Finnish article
                        #extract relevant NE from the Swedish article
                        text_sv = art_sv['content']
                        #subjects_sv = [s for s in subjects_fi if s in text_sv]
                        if 'subjects' in art_sv.keys():
                            subjects_sv2 = list(set(art_sv['subjects']).intersection(set(subjects_fi)))
                            subjects_sv = list(set(subjects_sv + subjects_sv2))
                        #check if the articles share 3 or more NEs
                        #inter = list(set(subjects_fi).intersection(set(subjects_sv)))
                        if len(subjects_sv2) >= 3:
                            if 'related_articles' not in art_fi.keys():
                                art_fi['related_articles'] = []
                            art_fi['related_articles'].append(art_sv)
                    elif day_delta >= 30:
                        break
        if 'related_articles' in art_fi:
            aligned_articles[art_fi['article_id']] = art_fi
    return aligned_articles


def align_articles_one_to_many_monthly(articles):
    aligned_articles = {}
    for art_fi in articles['fi']:
        print("Aligning Finnish article", art_fi['article_id'])
        date_fi = art_fi['date']
        if date_fi.year:
            if 'subjects' in art_fi.keys():
                subjects_fi = [s for s in art_fi['subjects'] if s is not None]
                for art_sv in articles['sv']:
                    date_sv = art_sv['date']
                    #day_delta = (art_sv['date'] - date_fi).days
                    if date_fi.year == date_sv.year and date_fi.month == date_sv.month: #check that Finnish and Swedish articles are in the same month
                        #extract relevant NE from the Swedish article
                        text_sv = art_sv['content']
                        subjects_sv = [s for s in subjects_fi if s in text_sv]
                        if 'subjects' in art_sv.keys():
                            subjects_sv2 = list(set(art_sv['subjects']).intersection(set(subjects_fi)))
                            subjects_sv = list(set(subjects_sv + subjects_sv2))
                        #check if the articles share 3 or more NEs
                        #inter = list(set(subjects_fi).intersection(set(subjects_sv)))
                        if len(subjects_sv2) >= 3:
                            if 'related_articles' not in art_fi.keys():
                                art_fi['related_articles'] = []
                            art_fi['related_articles'].append(art_sv)
        if 'related_articles' in art_fi:
            aligned_articles[art_fi['article_id']] = art_fi
    return aligned_articles




# link one Finnish news article to one or more Swedish articles
def align_articles_one_to_many2(articles):
    aligned_articles = {}
    aligned_dict = {}
    for art_fi in articles['fi']:
        print("Aligning Finnish article", art_fi['article_id'])
        date_fi = art_fi['date']
        if date_fi.year:
            if 'subjects' in art_fi.keys():
                subjects_fi = [s for s in art_fi['subjects'] if s is not None]
                for art_sv in articles['sv']:
                    day_delta = (art_sv['date'] - date_fi).days
                    if abs(day_delta) <= 5: #check Swedish articles published 2 days before/after the Finnish article
                        #extract relevant NE from the Swedish article
                        text_sv = art_sv['content']
                        subjects_sv = [s for s in subjects_fi if s in text_sv]
                        if 'subjects' in art_sv.keys():
                            subjects_sv2 = list(set(art_sv['subjects']).intersection(set(subjects_fi)))
                            subjects_sv = list(set(subjects_sv + subjects_sv2))
                        #check if the articles share 3 or more NEs
                        #inter = list(set(subjects_fi).intersection(set(subjects_sv)))
                        if len(subjects_sv) >= 3:
                            if 'related_articles' not in art_fi.keys():
                                art_fi['related_articles'] = []
                            art_fi['related_articles'].append(art_sv)
                            art_fi_id = art_fi['article_id']
                            if art_fi_id not in aligned_dict:
                                aligned_dict[art_fi_id] = []
                            aligned_dict[art_fi_id].append(art_sv['article_id'])
                    elif day_delta >= 30:
                        break
        if 'related_articles' in art_fi:
            aligned_articles[art_fi['article_id']] = art_fi
    return aligned_articles, aligned_dict


def write_articles_to_file(path):
    fp = open(path, "r")
    data = json.load(fp)
    languages = list(data.keys())
    text_data = {lang: {} for lang in languages}
    art_count = len(data['fi'])
    for i in range(art_count):
        print("Art count: ", i)
        date = data['fi'][i]['date']
        date = date.__str__()
        header = "||Article_id:" + str(i + 1)+"||"
        text_data_fi = data['fi'][i]['content']
        text_data_sv = data['sv'][i]['content']
        if date in text_data['fi'].keys():
            text_data['fi'][date] += "\n" + header + text_data_fi
            text_data['sv'][date] += "\n" + header + text_data_sv
        else:
            text_data['fi'][date] = header + text_data_fi
            text_data['sv'][date] = header + text_data_sv
    #write articles to text files
    dates = text_data['fi'].keys()
    parent_dir = "data/yle/raw_text/"
    for dat in dates:
        print("Date: ", dat)
        for lang in languages:
            fname = parent_dir+dat+"_"+lang+".txt"
            fp = open(fname, 'w')
            fp.write(text_data[lang][dat])
            fp.close()
            print("Saved file as: ", fname)
    print("Done writing all articles as raw text!")


def write_articles_to_file2(articles):
    languages = list(articles.keys())
    text_data = {lang: {} for lang in languages}
    for lang in languages:
        art_count = len(articles[lang])
        for i in range(art_count):
            print("Art count -",lang,":", i)
            article_id = articles[lang][i]['article_id']
            date = articles[lang][i]['date'].__str__()
            header = "||Article_id:" + article_id+"||"
            article_text_data = articles[lang][i]['content']
            if date in text_data[lang].keys():
                text_data[lang][date] += "\n" + header + article_text_data
            else:
                text_data[lang][date] = header + article_text_data
        #write articles to text files
        dates = text_data[lang].keys()
        parent_dir = "../data/yle/raw_text2/"
        for dat in dates:
            print("Date: ", dat)
            fname = parent_dir+dat+"_"+lang+".txt"
            with open(fname, 'w') as fp:
                fp.write(text_data[lang][dat])
                fp.close()
                print("Saved file as: ", fname)
    print("Done writing all articles as raw text!")


def write_aligned_articles_to_file(aligned_data, out_dir):
    languages = ['fi', 'sv']
    art_count_fi = 0
    art_count_sv = 0
    text_data = {lang: {} for lang in languages}
    for art_id in aligned_data:
        #print("FI article id:", art_id)
        date = aligned_data[art_id]['date'].__str__()
        if art_count_fi%10 == 0 and art_count_fi/10 > 0:
            date_count_fi = int(art_count_fi/10)
            date += "-" + str(date_count_fi)
        print("FI Date:", date)
        header = "||Article_id:" + str(art_id)+"||"
        article_text_data = aligned_data[art_id]['content']
        if date in text_data['fi'].keys():
            text_data['fi'][date] += "\n" + header + article_text_data
        else:
            text_data['fi'][date] = header + article_text_data
        art_count_fi += 1
        related_articles = aligned_data[art_id]['related_articles']
        for related_art in related_articles:
            art_id_rel = related_art['article_id']
            #print("SV article id:", art_id_rel)
            date_rel = related_art['date'].__str__()
            if art_count_sv % 10 == 0 and int(art_count_sv/10) > 0:
                date_count_sv = int(art_count_sv/10)
                date_rel += "-" + str(date_count_sv)
            print("SV Date: ", date_rel)
            header_rel = "||Article_id:" + str(art_id_rel) + "||"
            article_text_data_rel = related_art['content']
            if date_rel in text_data['sv'].keys():
                text_data['sv'][date_rel] += "\n" + header_rel + article_text_data_rel
            else:
                text_data['sv'][date_rel] = header_rel + article_text_data_rel
            art_count_sv += 1
    for lang in languages:
        print("Language: ", lang.upper())
        dates = text_data[lang].keys()
        parent_dir = "../data/yle/"+out_dir+"/"
        for dat in dates:
            print("Date: ", dat)
            fname = parent_dir+dat+"_"+lang+".txt"
            with open(fname, 'w') as fp:
                fp.write(text_data[lang][dat])
                fp.close()
                print("Saved file as: ", fname)
        print("Done writing all articles as raw text!")


def write_aligned_articles_to_file2(articles, valid_ids, out_dir):
    languages = ['fi', 'sv']
    text_data = {lang: {} for lang in languages}
    for lang in languages:
        art_count = 0
        for art in articles[lang]:
            art_id = art['article_id']
            #print("Art id:", art_id)
            if art_id in valid_ids:
                date = art['date']
                date_str = date.__str__()
                date_str += "-" + str(art_count-(art_count % 20))
                print("Date -", lang, ":", date_str)
                header = "||Article_id:" + str(art_id)+"||"
                article_text_data = art['content']
                if date_str in text_data[lang].keys():
                    text_data[lang][date_str] += "\n" + header + article_text_data
                else:
                    text_data[lang][date_str] = header + article_text_data
                art_count += 1
    for lang in languages:
        print("Language: ", lang.upper())
        dates = text_data[lang].keys()
        parent_dir = "../data/yle/"+out_dir+"/"
        for dat in dates:
            print("Date: ", dat)
            fname = parent_dir+dat+"_"+lang+".txt"
            with open(fname, 'w') as fp:
                fp.write(text_data[lang][dat])
                fp.close()
                print("Saved file as: ", fname)
        print("Done writing all articles as raw text!")

path = "../data/yle"
start_year = 2013
end_year = 2015
articles = process_articles(path=path, start_year=start_year, end_year=end_year)
aa = align_articles_one_to_many_monthly(articles)
outfile_pkl = "aligned_articles_"+str(start_year)+"_"+str(end_year)+"_monthly_2.pkl"
with open(outfile_pkl, "wb") as f:
    pickle.dump(aa, f)
    f.close()

for key in aa:
    aa[key]['date'] = aa[key]['date'].__str__()
    for rel in aa[key]['related_articles']:
        rel['date'] = rel['date'].__str__()

outfile = "aligned_articles_"+str(start_year)+"_"+str(end_year)+"_monthly_sv.json"
with open(outfile, 'w') as f:
    json.dump(aa, f)
    f.close()

print("***** Saved aligned articles as", outfile,"*****")
