import numpy as np

def _sample(cumsum_dist_list, rng):
    intervals = []
    for iterate in range(len(cumsum_dist_list)-1):
        intervals.append([cumsum_dist_list[iterate], cumsum_dist_list[iterate+1]])

    intervals = np.array(intervals)
    
    coin_toss_2 = np.random.uniform(0,1)
    index = np.where(np.logical_and(coin_toss_2 >= intervals[:,0], coin_toss_2 < intervals[:,1]) == True)[0][0]
    return index

#def sample_spec_gp(spec_prob, spec, top_k, seed):
def sample_spec_gp(spec_prob, spec, top_k, rng):

    
    if top_k > len(spec) :
        raise Exception("Top-K is greater than number of spcs present. Re-check implementation")
    elif top_k == len(spec):
        sampled_specs =spec
    else:
        sampled_specs = []

        while len(sampled_specs) != top_k:
            #print("spec",spec)
            #print("spec_prob",spec_prob)
            
            prob = [spec_prob[item] for item in spec]
            prob = [item / sum(prob) for item in prob]
            cumsum_dist_list_cl = np.insert(np.cumsum(prob),0,0)
            
            #index = _sample(cumsum_dist_list_cl, seed)
            index = _sample(cumsum_dist_list_cl, rng)
            sampled_specs.append(spec.pop(index))
           
         
    return sampled_specs

def sample_spec(spec_prob, unclassified_spec, classified_spec, top_k, classified_sample_bias, rng):
#def sample_spec(spec_prob, unclassified_spec, classified_spec, top_k, classified_sample_bias, seed):
    #print("classified_spec",classified_spec)
    #print("unclassified_spec",unclassified_spec)
    if top_k > len(classified_spec) + len(unclassified_spec):
        raise Exception("Top-K is greater than number of spcs present. Re-check implementation")
    elif top_k == len(classified_spec) + len(unclassified_spec):
        sampled_specs = unclassified_spec + classified_spec
    else:
        sampled_specs = []

        while len(sampled_specs) != top_k:
            
            if not unclassified_spec:
                # print("Case 1")
                prob = [spec_prob[item] for item in classified_spec]
                #print("prob1",prob)
                prob = [item / sum(prob) for item in prob]
                #print("prob2",prob)
                cumsum_dist_list_cl = np.insert(np.cumsum(prob),0,0)
                #print("cumsum_dist_list_cl",cumsum_dist_list_cl)
                #index = _sample(cumsum_dist_list_cl, seed)
                index = _sample(cumsum_dist_list_cl, rng)
                #print("index",index)
                sampled_specs.append(classified_spec.pop(index))
                #print("sampled_specs",sampled_specs)
            elif not classified_spec:
                # print("Case 1")
                interval_unclassified = len(unclassified_spec)
                cumsum_dist_list_uncl = np.insert(np.cumsum([1/interval_unclassified]*interval_unclassified),0,0)
                #index = _sample(cumsum_dist_list_uncl, seed)
                index = _sample(cumsum_dist_list_uncl, rng)
                sampled_specs.append(unclassified_spec.pop(index))
            else:

                coin_toss = np.random.uniform(0,1)
                if (coin_toss < classified_sample_bias):
                    # print("Case 3.1")
                    prob = [spec_prob[item] for item in classified_spec]
                    prob = [item / sum(prob) for item in prob]
                    cumsum_dist_list_cl = np.insert(np.cumsum(prob),0,0)
                    #index = _sample(cumsum_dist_list_cl, seed)
                    index = _sample(cumsum_dist_list_cl, rng)
                    sampled_specs.append(classified_spec.pop(index))
                else:
                    # print("Case 3.2")
                    interval_unclassified = len(unclassified_spec)
                    cumsum_dist_list_uncl = np.insert(np.cumsum([1/interval_unclassified]*interval_unclassified),0,0)
                    #index = _sample(cumsum_dist_list_uncl, seed)
                    index = _sample(cumsum_dist_list_uncl, rng)
                    sampled_specs.append(unclassified_spec.pop(index))
            # print(sampled_specs)
            # print(classified_spec)
            # print(unclassified_spec)
            # print("**************")
            # print("**************")
            # print("**************")
    return sampled_specs

