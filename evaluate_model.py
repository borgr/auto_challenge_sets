import codecs
import numpy as np
import subprocess
import scipy.stats as stats
import os

import matplotlib as mpl
mpl.use("agg")
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
mpl.style.use('classic')
rcParams.update({'figure.autolayout': True})
sns.set(font_scale=2)
sns.set_style("dark")
show = False

FAST_ALIGN_PATH = '/cs/snapless/oabend/borgr/SSMT/fast_align/build/fast_align'
DE_EN_FAST_MODEL = '/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/wmt16/fast_align_params.de-en'


def align(src_sents, trg_sents, alignments_file, model, reverse=False):
    assert(len(src_sents) == len(trg_sents)
           ), ("Wrong src ref sentence alignments", len(src_sents), len(trg_sents))
    alignment = ".revalignments" if reverse else ".alignments"
    output_file = alignments_file + alignment
    if os.path.isfile(output_file):
        print("reading alignment", output_file)
    else:
        alignments = [src + u" ||| " + trg for src,
                      trg in zip(src_sents, trg_sents) if src.strip() and trg.strip()]
        if not os.path.isdir(os.path.dirname(alignments_file)):
            os.makedirs(os.path.dirname(alignments_file))

        base_filename = alignments_file
        with codecs.open(base_filename + ".sentences", 'w', 'utf8') as f:
            f.write("\n".join(alignments))
            print("wrote to ", base_filename + ".sentences")
        reverse_flag = "-r" if reverse else ""
        command = FAST_ALIGN_PATH + " -d -o -v -i " + base_filename + \
            ".sentences" + " -f " + model + reverse_flag
        print(command)
        command = command.split()
        with open(output_file, "w") as fl:
            subprocess.check_call(command, stdout=fl)
        print("wrote to ", output_file)
    with codecs.open(output_file, "r", "utf8") as f:
        alignments = f.readlines()
    return alignments


def process_alignments(alignments):
    res = []
    for line in alignments:
        res.append(line.strip().split("|||"))
        # convert alignments to dict
        alignment = res[-1][2]
        alignment = alignment.split()
        alignment = [word_alignment.split("-") for word_alignment in alignment]
        alignment = {int(src): int(trg) for src, trg in alignment}
        assert len(alignment) == len(
            set((trg for src, trg in alignment.items()))), (len(alignment), len(set((trg for src, trg in alignment.items()))))
        for i in range(res[-1][0].strip().count(" ") + 1):
            if i not in alignment:
                alignment[i] = i
                # alignment[i] = -1
        res[-1][2] = alignment

        # convert fast align score to number
        res[-1][-1] = float(res[-1][-1])
    res = list(zip(*res))
    return res


def find_reordered_sentences(source_path, ref_path, align_dir, alignment_model, out_file=None, min_diff=5):
    with open(source_path) as fl:
        source = [line.strip() for line in fl]
    with open(ref_path) as fl:
        ref = [line.strip() for line in fl]
    ref_alignments = align(source, ref, os.path.join(
        align_dir, os.path.basename(source_path).split(".")[0] + "ref"), alignment_model, reverse=True)
    alignments = process_alignments(
        ref_alignments)
    diffs = []
    srcs = []
    outs = []
    for ref_src, ref_out, ref_alignment, ref_score in zip(*alignments):
        print(ref_src, ref_out, ref_alignment)
        diff = 0
        for src, trg in ref_alignment.items():
            cur_diff = abs(src - trg)
            if diff < cur_diff and trg >= 0 and ref_src[src].isalpha() and ref_out[trg].isalpha():
                diff = cur_diff
        diffs.append(diff)
        if diff > 10:
            print("______", "diff", diff, "______")
            print("src:", ref_src)
            print("ref:", ref_out)
        if diff > min_diff:
            srcs.append(ref_src)
            outs.append(ref_out)
    if out_file:
        with open(out_file + "_reorder" + str(min_diff) + ".de", "w") as fl:
            fl.write("\n".join(srcs))
        with open(out_file + "_reorder" + str(min_diff) + ".en", "w") as fl:
            fl.write("\n".join(outs))
    sns.distplot(diffs)
    plt.show()


def evaluate_order(source_path, ref_path, output_paths, align_dir, alignment_model):
    with open(source_path) as fl:
        source = [line.strip() for line in fl]
    with open(ref_path) as fl:
        ref = [line.strip() for line in fl]
    ref_alignments = align(source, ref, os.path.join(
        align_dir, "ref"), alignment_model, reverse=True)
    ref_srcs, ref_outs, ref_alignments, ref_scores = process_alignments(
        ref_alignments)
    for output_path in output_paths:
        with open(output_path) as fl:
            output = [line.strip() for line in fl]
        alignments_file = os.path.join(
            align_dir, os.path.basename(os.path.splitext(output_path)[0]))
        out_alignments = align(
            source, output, alignments_file, alignment_model, reverse=True)
        out_srcs, out_outs, out_alignments, out_scores = process_alignments(
            out_alignments)
        print(len(ref_alignments), len(out_alignments))
        scores = []
        for ref_alignment, out_alignment in zip(ref_alignments, out_alignments):
            ref_alignment = [ref_alignment[i]
                             for i in range(len(ref_alignment))]
            out_alignment = [out_alignment[i]
                             for i in range(len(out_alignment))]
            score = stats.kendalltau(ref_alignment, out_alignment)
            # if np.isnan(score[0]):
            #     print(score, ref_alignment, out_alignment)
            # print(score)
            scores.append(score)

        for score, src, ref, out in zip(scores, ref_srcs, ref_outs, out_outs):
            if (not np.isnan(score[0])) and score[0] < 0.8:
                print(score, src, ref, out)
        corrs = [score[0] for score in scores if not np.isnan(score[0])]
        sns.distplot(corrs)
        plt.show()

def main():
    import json
    res = {}
    with open("/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/vocab.clean.unesc.tok.tc.bpe.ende") as fl:
        for line in fl:
            res[line.strip()] = len(res) + 1
    with open("/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/vocab.clean.unesc.tok.tc.bpe.ende.json", "w") as fl:
        json.dump(res, fl, indent=0)

    align_dir = "/cs/snapless/oabend/borgr/locality/calc"

    ref = "/cs/snapless/oabend/borgr/locality/data/newstest2013.unesc.tok.en"
    source = "/cs/snapless/oabend/borgr/locality/data/newstest2013.unesc.tok.de"
    # outputs = [
    #     "/cs/snapless/oabend/borgr/locality/data/newstest2013.unesc.tok.tc.en.output_model_epoch_8_step_80000"]
    # evaluate_order(source, ref, outputs, align_dir, DE_EN_FAST_MODEL)
    # challenge_dir = r"/cs/snapless/oabend/borgr/locality/data/challenge"
    # find_reordered_sentences(source, ref, align_dir, DE_EN_FAST_MODEL, challenge_dir)
if __name__ == '__main__':
    main()
