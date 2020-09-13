# auto_challenge_sets
Automatically Extracting Challenge Sets for Non-local Phenomena in Neural Machine Translation git repo

Please cite if you find it useful:
@inproceedings{choshen-abend-2019-automatically,
    title = "Automatically Extracting Challenge Sets for Non-Local Phenomena in Neural Machine Translation",
    author = "Choshen, Leshem  and
      Abend, Omri",
    booktitle = "Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/K19-1028",
    doi = "10.18653/v1/K19-1028",
    pages = "291--303",
    abstract = "We show that the state-of-the-art Transformer MT model is not biased towards monotonic reordering (unlike previous recurrent neural network models), but that nevertheless, long-distance dependencies remain a challenge for the model. Since most dependencies are short-distance, common evaluation metrics will be little influenced by how well systems perform on them. We therefore propose an automatic approach for extracting challenge sets rich with long-distance dependencies, and argue that evaluation using this methodology provides a complementary perspective on system performance. To support our claim, we compile challenge sets for English-German and German-English, which are much larger than any previously released challenge set for MT. The extracted sets are large enough to allow reliable automatic evaluation, which makes the proposed approach a scalable and practical solution for evaluating MT performance on the long-tail of syntactic phenomena.",
}
