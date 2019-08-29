import os
import evaluate_model as em
from random import shuffle
from collections import Counter
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import counts_data
mpl.style.use('classic')

de_udpipe_conllu_path = r"/cs/snapless/oabend/borgr/SSMT/log/generator_model-16-5.8-lr-0.0001/train.clean.unesc.tok.tc.de.1000000.conllu"
out_dir = r"/cs/snapless/oabend/borgr/locality/data/challenge"


def count_head_diff(conllu_path, labels_subset=None):
    distances = {}
    processed = 0
    with open(conllu_path) as fl:
        for line in fl:
            line = line.strip()
            if line and not line.startswith("#"):
                split = line.split()
                if len(split) == 10:
                    idx, form, lemma, upos, xpos, feats, head, deprel, deps, misc = split
                    idx = int(idx)
                    head = int(head)
                else:
                    print("line of unexpected form", line)
                if (not labels_subset) or deprel in labels_subset:
                    if deprel not in distances:
                        distances[deprel] = []
                    distances[deprel].append(idx - head)
            elif "sent_id" in line:
                processed += 1
                if processed % 100000 == 0:
                    print(processed, "sentences processed")
                    print(sorted([(deprel, np.mean(dist))
                                  for deprel, dist in distances.items()]))
                    print(sorted([(deprel, len(dist))
                                  for deprel, dist in distances.items()]))
    return distances


def particle_condition(parse, parse_head):
    return "prt" in parse[7].lower()


def reflexive_condition(parse, parse_head):
    return "Reflex=Yes".lower() in parse[5].lower()


def preposition_stranding_condition(parse, parse_head):
    return "obl" in parse[7].lower() and parse[3] == "ADP"


def extract_distant_heads(conllu_path, out_path, func=lambda x, y: True, min_dist=2, exact_distance=False):
    print("Extracting", out_path)
    processed = 0
    doc = -1
    doc_size = 10000
    with open(conllu_path) as fl:
        sentence = []
        parse = ""
        for line in fl:
            parse += line
            line = line.strip()
            if line.strip() and not line.startswith("#"):
                split = line.split()
                if len(split) == 10:
                    cur_splits.append(split)
                    idx, form, lemma, upos, xpos, feats, head, deprel, deps, misc = split
                    idx = int(idx)
                    head = int(head)
                    if abs(idx - head) > min_dist and ((not exact_distance) or abs(idx - head) == min_dist + 1):
                        check_tuples.append((idx, head))
                        check_sentence = True
                    sentence += [(idx, form)]
                else:
                    print("line of unexpected form", line)
            elif "newdoc" in line:
                doc += 1
            elif "sent_id" in line:
                sentence = []
                parse = line
                check_tuples = []
                cur_splits = []
                add_sentence = False
                check_sentence = False
                cur_id = int(line.split("=")[-1])
                processed += 1
                if processed % 10000 == 0:
                    print("processed ", processed)
            elif (not line.strip()) and check_sentence:
                conds = [func(cur_splits[node - 1], cur_splits[head - 1])
                         for node, head in check_tuples]
                if any(conds):
                    assert min([abs(node - head)
                                for node, head in check_tuples]) > min_dist

                    sentence_text = " ".join([form for idx, form in sentence])

                    mark = set(np.array(check_tuples)[conds].flatten())
                    marked_sentence = [(tmp_idx, "*" + form + "*") if tmp_idx in mark else (tmp_idx, form) for (tmp_idx, form)
                                       in sentence]
                    marked_sentence_text = " ".join(
                        [form for (idx, form) in marked_sentence])
                    _writeline_distant(
                        sentence_text, marked_sentence_text, parse, cur_id + doc_size * doc, out_path)


def _writeline_distant(sentence, marked_sentence, parse, cur_id, out_path):
    ids_path = out_path + ".ids"
    sentences_path = out_path + ".txt"
    marked_sentences_path = out_path + "_marked.txt"
    conllu_path = out_path + ".conllu"
    # print("writes" , cur_id)
    with open(ids_path, "a+") as fl:
        fl.write(str(cur_id))
        fl.write("\n")
    with open(sentences_path, "a+") as fl:
        fl.write(sentence)
        fl.write("\n")
    with open(marked_sentences_path, "a+") as fl:
        fl.write(marked_sentence)
        fl.write("\n")
    with open(conllu_path, "a+") as fl:
        fl.write(parse)
        fl.write("\n")


def sample_lines(path, ids, out_path):
    cur_id = 0
    bias = -1
    ids = list(ids)
    with open(path) as in_fl:
        with open(out_path, "w+") as out_fl:
            for i, line in enumerate(in_fl):
                if i == ids[cur_id] + bias:
                    # print(i)
                    out_fl.write(line)
                    cur_id += 1
                    if cur_id == len(ids):
                        return
                    # print(cur_id, len(ids))
                    if ids[cur_id] < ids[cur_id - 1]:
                        bias += 10000
                        # print("bias", bias)


def create_parallel(src, trg, ids, out_path):
    sample_lines(src, ids, out_path + ".de")
    sample_lines(trg, ids, out_path + ".en")


def create_trial_filename(trial_name, min_dist, exact_distance):
    min_sym = "" if min_dist == 1 else str(min_dist)
    exact_sym = "_exactly" if exact_distance else ""
    return trial_name + exact_sym + min_sym

if __name__ == '__main__':
    ## example of use of extraction of reordering and lexical LDDs
    align_dir = "/cs/snapless/oabend/borgr/locality/calc"
    source = "/cs/snapless/oabend/borgr/locality/data/newstest2013.unesc.tok.de"
    ref = "/cs/snapless/oabend/borgr/locality/data/newstest2013.unesc.tok.en"
    en_news_conllu = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/21.1/UD/newstest2013.en.conllu"
    # em.find_reordered_sentences(
    #     source, ref, align_dir, em.DE_EN_FAST_MODEL, os.path.join(out_dir, "newstest2013"))
    # min_dists = [1]
    # exact_distance = True
    # for min_dist in min_dists:
    #     trial_name = "en_preposition_stranding" + "_news"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(en_news_conllu, os.path.join(
    #         out_dir, trial_name), preposition_stranding_condition, min_dist, exact_distance=exact_distance)
    #     trial_name = "en_particle" + "_news"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(en_news_conllu, os.path.join(
    #         out_dir, trial_name), particle_condition, min_dist, exact_distance=exact_distance)

    #     trial_name = "en_reflexive" + "_news"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(en_news_conllu, os.path.join(
    #         out_dir, trial_name), reflexive_condition, min_dist, exact_distance=exact_distance)

    #     de_news_conllu = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/21.1/UD/newstest2013.de.conllu"

    #     trial_name = "de_particle" + "_news"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(de_news_conllu, os.path.join(
    #         out_dir, trial_name), particle_condition, min_dist, exact_distance=exact_distance)

    #     trial_name = "de_reflexive" + "_news"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(de_news_conllu, os.path.join(
    #         out_dir, trial_name), reflexive_condition, min_dist, exact_distance=exact_distance)

    #     trial_name = "de_preposition_stranding" + "_news"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(de_news_conllu, os.path.join(
    #         out_dir, trial_name), preposition_stranding_condition, min_dist, exact_distance=exact_distance)

    # for root, dirs, filenames in os.walk(out_dir):
    #     for filename in filenames:
    #         if filename.endswith(".ids") and "_news" in filename:
    #             path = os.path.join(root, filename)
    #             with open(path) as fl:
    #                 ids = [int(i) for i in fl]
    #             create_parallel(source, ref, ids, os.path.splitext(path)[0])

    source = "/cs/snapless/oabend/borgr/SSMT/data/en_de/Books.de-en.de"
    ref = "/cs/snapless/oabend/borgr/SSMT/data/en_de/Books.de-en.en"
    # # source_bpe = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/21.1/Books.de-en.unesc.tok.en"
    # # ref_bpe =
    # # "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/21.1/Books.de-en.unesc.tok.bpe.de"

    # em.find_reordered_sentences(
    #     source, ref, align_dir, em.DE_EN_FAST_MODEL, os.path.join(out_dir, "Books"))
    # en_books_conllu = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/21.1/UD/Books.de-en.en.conllu"
    # min_dists = [0, 1, 2, 3, 4, 5]
    # for min_dist in min_dists:
    #     trial_name = "en_preposition_stranding"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(en_books_conllu, os.path.join(
    #         out_dir, trial_name), preposition_stranding_condition, min_dist, exact_distance=exact_distance)
    #     trial_name = "en_particle"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(en_books_conllu, os.path.join(
    #         out_dir, trial_name), particle_condition, min_dist, exact_distance=exact_distance)

    #     trial_name = "en_reflexive"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(en_books_conllu, os.path.join(
    #         out_dir, trial_name), reflexive_condition, min_dist, exact_distance=exact_distance)

    #     de_books_conllu = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/21.1/UD/Books.de-en.de.conllu"

    #     trial_name = "de_particle"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(de_books_conllu, os.path.join(
    #         out_dir, trial_name), particle_condition, min_dist, exact_distance=exact_distance)

    #     trial_name = "de_reflexive"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(de_books_conllu, os.path.join(
    #         out_dir, trial_name), reflexive_condition, min_dist, exact_distance=exact_distance)

    #     trial_name = "de_preposition_stranding"
    #     trial_name = create_trial_filename(
    #         trial_name, min_dist, exact_distance)
    #     extract_distant_heads(de_books_conllu, os.path.join(
    #         out_dir, trial_name), preposition_stranding_condition, min_dist, exact_distance=exact_distance)

    for root, dirs, filenames in os.walk(out_dir):
        for filename in filenames:
            if filename.endswith(".ids") and "_news" not in filename:
                path = os.path.join(root, filename)
                with open(path) as fl:
                    ids = [int(i) for i in fl]
                create_parallel(source, ref, ids, os.path.splitext(path)[0])
    # Extract subset of the sentences
    sentences_num = 10000
    create_parallel(source, ref, list(range(5,sentences_num+5)), os.path.join(root, os.path.splitext(os.path.basename(source))[0] + "_" + str(sentences_num)))
                # create_parallel(source_bpe, ref_bpe, ids,
                # os.path.splitext(path)[0] + ".bpe")

    # distances = count_head_diff(de_udpipe_conllu_path)
    # print("occurrences", sorted([(deprel, len(dist))
    #                              for deprel, dist in distances.items()]))
    # print("mean dist", sorted([(deprel, np.mean(dist))
    #                            for deprel, dist in distances.items()]))
    # print("mean abs dist", sorted(
    #     [(deprel, np.mean(np.abs(dist))) for deprel, dist in distances.items()]))
