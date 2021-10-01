#!/bin/bash
#SBATCH --mem=30g
#SBATCH -c4
#SBATCH --time=10:0:0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/labs/oabend/lovodkin93/encoder_masking_data/pre_process_same_scene_mask/slurm/preprocess_challenge-%j.out

source /cs/snapless/oabend/borgr/envs/tg/bin/activate

# this sample script preprocesses a sample corpus, including tokenization,
# truecasing, and subword segmentation.
# for application to a different language pair,
# change source and target prefix, optionally the number of BPE operations,
# source /cs/snapless/oabend/borgr/envs/SSMT/bin/activate
# source /cs/snapless/oabend/borgr/envs/non_proficient/bin/activate
script_dir=`dirname $0`
main_dir=$script_dir/..
data_dir=/cs/labs/oabend/lovodkin93/encoder_masking_data/pre_process_same_scene_mask/en_tr/data/challenge/seperated
src=en
trg=tr
# scripts directory of moses decoder: http://www.statmt.org/moses/
# you do not need to compile moses; a simple download is sufficient
#moses_scripts=/home/bhaddow/moses.new/dist/977e8ea/scripts
moses_scripts=/cs/snapless/oabend/borgr/SSMT/preprocess/mosesdecoder/scripts

#scripts for subword segmentation: https://github.com/rsennrich/subword-nmt
#bpe_scripts=/home/bhaddow/tools/subword-nmt
bpe_scripts=/cs/snapless/oabend/borgr/SSMT/preprocess/subword-nmt

filter_lines="filter_lines.py"
# filtering charateristics
min_length=1
max_length=1000
ratio=5
alignment_score="-180"
fast_align=/cs/snapless/oabend/borgr/SSMT/fast_align/build/fast_align

nematus_home=/cs/snapless/oabend/borgr/TG/

#jieba word segmentation utility: https://pypi.python.org/pypi/jieba/
#this is only required for Chinese
zh_segment_home=/mnt/baldur0/tramooc/tools/jieba

unescape=/cs/snapless/oabend/borgr/SSMT/preprocess/unescape.py

##minimum number of times we need to have seen a character sequence in the training text before we merge it into one unit
##this is applied to each training text independently, even with joint BPE
#bpe_threshold=50

src_model_dir=/cs/labs/oabend/lovodkin93/encoder_masking_data/pre_process_same_scene_mask/en_tr/preprocess/model/06.06.21/
trg_model_dir=/cs/labs/oabend/lovodkin93/encoder_masking_data/pre_process_same_scene_mask/en_tr/preprocess/model/06.06.21/
out_data_dir=/cs/labs/oabend/lovodkin93/encoder_masking_data/pre_process_same_scene_mask/en_tr/preprocess/06.06.21/challenge/
if [ ! -d $out_data_dir ]; then
  mkdir $out_data_dir
fi
# tokenize
# for prefix in Books_reorder5 newstest2013
pattern=".*unesc.*"
marked=".*mark.*"
has_digits="[0-9]"
for filename in $data_dir/*;
 do
   filename=$(basename -- "$filename")
   prefix="${filename%.*}"
   if [[ $prefix =~ $pattern ]] || [[ $prefix =~ $has_digits ]] || [[ $prefix =~ marked ]] || [ -f $out_data_dir/$prefix.tok.$src ]; then
      : # echo "File found ${out_data_dir}/${prefix}.tok.$src, skipping preprocessing."
   else
    echo "Processing $prefix"

    cp $data_dir/$prefix.$src $out_data_dir/$prefix.cln.$src
    cp $data_dir/$prefix.$trg $out_data_dir/$prefix.cln.$trg
#    python /cs/snapless/oabend/borgr/SSMT/preprocess/preprocess.py -t $trg -s $src -f $data_dir -n "$prefix" -o "$prefix.cln" -d "$out_data_dir"
#    python $unescape $trg
    cat $out_data_dir/$prefix.cln.$src | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
    $moses_scripts/tokenizer/tokenizer.perl -no-escape -a -l $src > $out_data_dir/$prefix.tok.$src

    cat $out_data_dir/$prefix.cln.$trg | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $trg | \
    $moses_scripts/tokenizer/tokenizer.perl -no-escape -a -l $trg > $out_data_dir/$prefix.tok.$trg

    #apply filtering
    python3 $filter_lines $out_data_dir/$prefix.clean $src $trg $out_data_dir/$prefix.tok.$src $out_data_dir/$prefix.tok.$trg --ratio $ratio --min $min_length --max $max_length --fast_align $fast_align --fast_score $alignment_score
    rm -rf $out_data_dir/tmp

    # apply truecaser (dev/test files)

    $moses_scripts/recaser/truecase.perl -model $src_model_dir/truecase-model.train.$src < $out_data_dir/$prefix.clean.$src > $out_data_dir/$prefix.tc.$src
    $moses_scripts/recaser/truecase.perl -model $trg_model_dir/truecase-model.train.$trg < $out_data_dir/$prefix.clean.$trg > $out_data_dir/$prefix.tc.$trg

    # apply BPE for joint bpes

    $bpe_scripts/apply_bpe.py -c $src_model_dir/$src$trg.model --glossaries "=" < $out_data_dir/$prefix.tc.$src > $out_data_dir/$prefix.bpe.$src
    $bpe_scripts/apply_bpe.py -c $trg_model_dir/$src$trg.model --glossaries "=" < $out_data_dir/$prefix.tc.$trg > $out_data_dir/$prefix.bpe.$trg

#    # apply BPE for separate bpes
#
#    $bpe_scripts/apply_bpe.py -c $src_model_dir/bpe.model.$src --glossaries "=" < $out_data_dir/$prefix.tc.$src > $out_data_dir/$prefix.bpe.$src
#    $bpe_scripts/apply_bpe.py -c $trg_model_dir/bpe.model.$trg --vocabulary $out_data_dir/../vocab.clean.unesc.tok.tc.bpe.$trg < $out_data_dir/$prefix.tc.$trg > $out_data_dir/$prefix.bpe.$trg
   fi
 done
if [ ! -f "$out_data_dir/preprocess_challenges_pretrained.sh" ]; then
  cp $0 $out_data_dir/preprocess_challenges_pretrained.sh
fi
# # build network dictionary
# $nematus_home/data/build_dictionary.py $out_data_dir/corpus.bpe.$src $out_data_dir/corpus.bpe.$trg

