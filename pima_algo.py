import time
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list


# amat :=       alignment matrix with shape (N,L,2); last dimension is
#               part of an unused feature, so ignore it
# namat :=      list of alignment iterations; each entry is a dictionary
#               with the iteration's amat, score, method, and time;
#               pima_init() initializes the list from the traces, and
#               pima_iter() automatically iterates from last namat entry
# traces :=     list of log's traces; each trace is a numpy array of
#               integers; the integers map to activity types


# csv to log, returns list of traces and case and activity index mappings
def csv2log(filename, caseids_col, acts_col,
            starts_col, isplg=False):
    # csv to df
    df = pd.read_csv(filename)
    if isplg:
        for id in df[caseids_col]:
            df[caseids_col] = int(id[9:])
    df = df.sort_values(by=[caseids_col, starts_col])

    # index unique caseIDs and activities
    caseids = df[caseids_col].as_matrix()
    ucaseids = np.unique(caseids)
    case_dict = dict(zip(range(0,ucaseids.size), ucaseids))

    acts = df[acts_col].as_matrix()
    act2id = dict(zip(np.unique(acts), range(1,acts.size+1)))
    act_dict = dict(zip(range(1,acts.size+1), np.unique(acts)))
    act_dict[0] = '-'

    # construct log
    traces = []
    for case in ucaseids:
        case_acts = acts[caseids==case]
        traces.append(np.asarray([act2id[act]
                for act in case_acts], dtype='int64'))

    # return
    return traces, case_dict, act_dict

# alignment scoring
def score(amat, metric):
    if len(amat.shape)>2: 
        amat = amat[:,:,0]
    if metric == 'length':
        return amat.shape[1]
    elif metric == 'avg_freq':
        return float(np.sum(amat!=0))/amat.size
    elif metric == 'sum_pair':
        freq = np.sum(amat!=0, axis=0)
        return np.sum(freq*(amat.shape[0]-freq))
# metric 'report' that is saved to 'score' in namat dictionaries
repo = ['length','sum_pair',]
# returns log as list of traces without gaps
def dealign(traces):
    return [t[t[:,0]!=0] for t in traces]
# returns matrix with empty columns removed
def degap(amat):
    return amat[:,np.max(amat, axis=0)[:,0]!=0,:]

# returns indices of traces sorted by a metric
def amat_argsort(amat, metric, order='ascending'):
    if len(amat.shape)>2:
        amat = np.squeeze(amat[:,:,0])
    if metric=='number':
        order = np.argsort(np.sum(amat!=0,axis=1))
    elif metric=='type':
        order = np.argsort(np.sum(amat,axis=1))
    if order[0]=='d':
        order = np.flip(order)
    return order

# NW pairwise alignment with modified sps objective function
def align_pair(a, b, score_only=False):
    # preprocess traces
    if len(a.shape)==1:                             ### the first iteration
        a = a[np.newaxis,:,np.newaxis]
        a = np.concatenate((a,a!=0), axis=2)
    elif len(a.shape)==2:                           ### the 2,3,4,... th iteration
        a = a[:,:,np.newaxis]
        a = np.concatenate((a,a!=0), axis=2)        ### [[[10.  1.]
                                                    ###   [9.   1.]
                                                    ###   [0.   0.]]]
                                                    ### shape = (trace#: N, trace length: L, 2)
    if len(b.shape)==1:
        b = b[np.newaxis,:,np.newaxis]
        b = np.concatenate((b,b!=0), axis=2)
    elif len(b.shape)==2:
        b = b[:,:,np.newaxis]
        b = np.concatenate((b,b!=0), axis=2)

    # flip traces
    a = np.flip(a, 1)
    b = np.flip(b, 1)

    # set up for alignment
    atr, alen, _ = a.shape
    btr, blen, _ = b.shape
    ntr = atr + btr                                 ### # of traces
    weigh_a = np.sum(a[:,:,1], axis=0)              ### count col activities
    weigh_b = np.sum(b[:,:,1], axis=0)              ### [6, 5, 10,...,...]
    acts_a = np.max(a[:,:,0], axis=0)               ### col activity
    acts_b = np.max(b[:,:,0], axis=0)               ### [16, 15, 9, 6,...,...]:
    # if alen < 30:
    #     print acts_a
    # print weigh_a
    # print alen, blen
    
    # initialize alignment matrix
    amat = np.zeros((alen + 1, blen + 1), dtype='float64')
    amat[1:, 0] = np.cumsum(-weigh_a*(ntr-weigh_a))
    amat[0, 1:] = np.cumsum(-weigh_b*(ntr-weigh_b))

    # initialize path matrix
    if not score_only:
        pmat = np.zeros((alen + 1, blen + 1), dtype='int8')
        pmat[1:, 0] = 1
        pmat[0, 1:] = -1

    # calculate alignment and path matrices
    minval = np.finfo(amat.dtype).min               ### min value of type float64 
    for i in range(0, alen):
        for j in range(0, blen):
            # calculate candidates                  ### 
            wa = weigh_a[i]; wb = weigh_b[j]
            match = amat[i,j] - (wa+wb)*(ntr-wa-wb)\
                if acts_a[i]==acts_b[j] else minval
            ins_a = amat[i,   j+1] - wa*(ntr-wa)
            ins_b = amat[i+1, j  ] - wb*(ntr-wb)

            # set alignment matrix value
            val = max(match, ins_a, ins_b)
            amat[i+1,j+1] = val

            # set path matrix value
            if not score_only:
                if val == match:
                    pmat[i+1,j+1] = 0
                elif val == ins_a:
                    pmat[i+1,j+1] = 1
                elif val == ins_b:
                    pmat[i+1,j+1] = -1

    # calculate score
    score = amat[-1, -1]
    if score_only: return score

    # calculate path
    path = np.asarray([], dtype='int8')
    i = alen
    j = blen
    while (i > 0) | (j > 0):
        t = pmat[i, j]
        path = np.append(path, t)
        if t == 1:
            i -= 1
        elif t == -1:
            j -= 1
        else:
            i -= 1
            j -= 1

    # combine traces
    c = np.zeros((atr + btr, path.size, 2), dtype='float64')
    i = alen - 1
    j = blen - 1
    for t in range(0, path.size):
        if path[t] == 1:
            c[:atr, t] = a[:, i]      ### row 0 to atr  = column i
            i -= 1
        elif path[t] == -1:
            c[atr:, t] = b[:, j]      ### row atr to end = column j
            j -= 1
        else:
            c[:atr, t] = a[:, i]
            c[atr:, t] = b[:, j]
            i -= 1
            j -= 1

    # return score and alignment
    # print score
    return score, c

# pima initialization
def pima_init(traces, method, repo=repo, verbose=True):
    totalt = time.time()
    # (random, order, seed)
    if method[0] == 'random':
        # build tree
        buildt = time.time()
        np.random.seed(method[2])               ### method = ('random','shuffle',seed); seed = 0;
        order = np.arange(len(traces))          ### order = array([0, 1, 2, ... , L-1])
        np.random.shuffle(order)                ### order = array([2, 5, 1, 0, ... ])
        buildt = time.time() - buildt

        # merge
        exect = time.time()
        if method[1] == 'shuffle':
            tmat = traces[order[0]]             ### tmat = trace0 (a random trace); [10 4 16 16 16 3 1 ...]
            for i in range(1, len(traces)):
                _, tmat = align_pair(tmat, traces[order[i]]) ### align_pair(random trace, another random trace)
        exect = time.time() - exect

        # reorder
        amat = np.zeros(tmat.shape, dtype=tmat.dtype)
        amat[order] = tmat

    # score initialization
    scrt = time.time()
    scrs = dict([(s, float(score(amat, s))) for s in repo])
    print(scrs)
    scrt = time.time() - scrt

    # wrap up
    totalt = time.time() - totalt
    if verbose:
        print('INITIALIZATION')
        print(str(scrs))
    return [{'amat': amat,
             'score': scrs,
             'method': method,
             'time': {'total': totalt,
                      'build': buildt,
                      'merge': exect,
                      'score': scrt}}]

# pima iterations
def pima_iter(namat, method, repo=repo, verbose=True):
    totalt = time.time()

    # unpack last amat
    amat = namat[-1]['amat']
    
    # (tracewise, order, seed)
    if method[0] == 'tracewise':
        # build order
        buildt = time.time()
        if method[1] == 'random':
            ix = np.arange(0, amat.shape[0])        ### number of traces
            np.random.seed(method[2])
            np.random.shuffle(ix)
        elif method[1] == 'given':
            ix = method[2]
        buildt = time.time() - buildt

        # merge
        exect = time.time()
        for i in ix:
            # split and remove gaps
            a = np.asarray([amat[i]])               ### random trace from amat
            b = np.delete(amat, i, axis=0)          ### amat - trace a
            a = degap(a)
            b = degap(b)
            
            # merge
            _, amat = align_pair(a, b)

            # reorder final alignment
            a = amat[0]
            b = np.delete(amat, 0, axis=0)
            amat = np.insert(b, i, a, axis=0)
        exect = time.time() - exect

    # (columnwise, criteria, threshold, order)
    elif method[0] == 'columnwise':
        # collect repeat candidates
        buildt = time.time()
        acts = np.max(amat[:, :, 0], axis=0)
        occs = np.zeros(int(np.max(acts) + 1))                  ### [0, 0, 0, 0, ...]
        for a in acts: occs[int(a)] += 1                        ### [0, 1, 3, 2, ...] occurance
        cand = np.asarray([occs[int(a)] > 1 for a in acts])     ### occurance > 1 [false, false, true, true , ...]

        # build order
        if method[1] == 'range':
            freq = np.mean(amat[:, :, 0] != 0, axis=0)          ### [0.5, 0, 1, 0.8, 0, .. .] acts freq
            lo, hi = method[2]                                  ### (0.1,0.9)
            chosen = np.where(cand & (freq != 1) &              ### [13  19  23  29  40  44  47  50  52  53  61  66  73  85  90  95  96 104 108]
                              (lo <= freq) & (freq <= hi))[0]
            ords = [(i, np.sum(amat[:, i, 0] != 0),             ### ords = [(index, # of acts in col i, array([index of acts !=0, index of acts == 0]))] 
                     np.concatenate((
                         np.where(amat[:, i, 0] != 0)[0],
                         np.where(amat[:, i, 0] == 0)[0])))
                    for i in chosen]
            ords = sorted(ords, key=lambda x: x[1],             ### 
                          reverse=method[3][0] == 'd')          ### descending order: # of acts in col i
        buildt = time.time() - buildt
        print(ords)

        # merge
        exect = time.time()
        for merge in ords:
            i, cutoff, ording = merge                           ### cutoff == # of acts in col i ??

            # split and remove gaps
            a = amat[ording[0:cutoff], :, :]
            b = amat[ording[cutoff:], :, :]
            a = degap(a)
            b = degap(b)

            # merge
            _, tmat = align_pair(a, b)

            # reorder final alignment
            amat = np.zeros(tmat.shape)
            amat[ording, :, :] = tmat
        exect = time.time() - exect

    # score initialization
    scrt = time.time()
    scrs = dict([(s, float(score(amat, s))) for s in repo])
    scrt = time.time() - scrt

    # wrap up
    totalt = time.time() - totalt
    namat.append({'amat': amat,
                  'score': scrs,
                  'method': method,
                  'time': {'total': totalt,
                           'build': buildt,
                           'merge': exect,
                           'score': scrt}})
    if verbose:
        print('ITERATION ' + str(len(namat) - 1))
        print(str(namat[-1]['score']))
    return namat

# find consensus sequence
def find_cs(amat, act_dict):
    col_act = np.amax(amat, axis = 0)
    cs = []
    for i in range(col_act.shape[0]):
        cs.append(act_dict[col_act[i]])
    return cs

# main function
def main():
    # EXAMPLE USAGE: CONVERGE TWICE
    # LOAD DATA
    log = csv2log(filename='Synthetic_activityTraces_1000.csv',
                  caseids_col='caseID',
                  acts_col='activity', 
                  starts_col='startTime')


    (traces, case_dict, act_dict) = log
    seed = 0
    # INITIALIZATION with random sequential method
    namat = pima_init(traces, ('random','shuffle',seed)) 
    print(namat)
    # alternate initialization with existing edit-distance methods
    # namat = pima_init(traces, ('tree','edit','single'))

    # SINGLE-TRACE ITERATIONS with random sequential method (1st convergence)
    while True:
        # perform iteration
        namat = pima_iter(namat, ('tracewise','random',seed+len(namat)))
        # define convergence condition
        prev = namat[-2]['score']['sum_pair']
        next = namat[-1]['score']['sum_pair']
        if (prev - next) / prev <= 0.00:
            break

    # MULTI-TRACE ITERATION with range (0.1,0.9) by descending frequency
    namat = pima_iter(namat, ('columnwise','range',(0.1,0.9),'des'))

    # SINGLE-TRACE ITERATIONS with random sequential method (2nd convergence)
    while True:
        # perform iteration
        namat = pima_iter(namat, ('tracewise', 'random', seed+len(namat)))

        # define convergence condition
        prev = namat[-2]['score']['sum_pair']
        next = namat[-1]['score']['sum_pair']
        if (prev - next) / prev <= 0.00:
            break

    # RESULT
    amat = namat[-1]['amat'][:,:,0]

    # FIND consensus sequence
    consensus_sequence = find_cs(amat, act_dict)

    # REPLACE the values in alignment matrix
    df_amat = pd.DataFrame(amat)
    for key in act_dict:
    	df_amat.replace(key, act_dict[key], inplace = True) 

    print(df_amat)

    # save results
    df_amat.to_csv('alignment.csv')
    import csv
    with open('consensus_sequence.csv', 'w') as csvfile:
    	cw = csv.writer(csvfile, delimiter = ',')
    	for act in consensus_sequence:
    		cw.writerow([act, 0])

if __name__ == "__main__":
    main()