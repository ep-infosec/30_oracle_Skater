import numpy as np
from skater.util.text_ops import cleaner


def _handling_ngrams_wts(original_feat_dict):
    # Currently, when feature dictionary contains continuous sequences such as 2-gram/3-gram etc. sequences as features,
    # a short term solution is to further split them into additional keys.
    # e.g. {'stay ball dropped': 0.5} will be split into a new dict as {'stay':0.5, 'ball': 0.5, 'dropped': 0.5}
    # TODO: this is just a temporary solution for handling n-grams, figure out a better solution
    for k in list(original_feat_dict.keys()):
        additional_keys = k.split()
    for a_k in additional_keys:
        if a_k in original_feat_dict:
            original_feat_dict[a_k] += original_feat_dict[a_k]
        else:
            original_feat_dict[a_k] = original_feat_dict[k]
    new_dict = original_feat_dict
    return new_dict


def relevance_wt_assigner(raw_txt, wts_as_dict):
    # normalize score by absolute max value
    if isinstance(wts_as_dict, dict):
        feature_scores = _handling_ngrams_wts(wts_as_dict)
        max_wt = np.max(np.abs(list(feature_scores.values())))
        wts_as_dict = {word: feature_scores[word] / max_wt for word in feature_scores.keys()}
        # transform dict into list of tuples (word, relevance_wts)
        # TODO: look into removing the below occurring side effect
        relevance_wts = []
        for word in raw_txt.split():
            # Clean up the raw word for irregularities
            word_cleaned_as_key = cleaner(word)
            if word_cleaned_as_key in wts_as_dict.keys():
                relevance_wts.append((word, wts_as_dict[word_cleaned_as_key]))
            else:
                relevance_wts.append((word, None))
    else:
        raise Exception('relevance wts currently needs to be as dict')
    return relevance_wts
