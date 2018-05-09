import numpy as np
import pandas as pd
import math
import random
import pima_algo as ali
import graphviz as gv

np.set_printoptions(threshold=np.nan)
# csv to log, returns list of traces and dictionaries
def csv2log(filename, caseids_col, acts_col, starts_col, isplg=False):
    df = pd.read_csv(filename)
    if isplg:
        for id in df[caseids_col]:
            df[caseids_col] = int(id[9:])
    df = df.sort_values(by=[caseids_col,starts_col])

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
        traces.append(np.asarray([act2id[act] for act in case_acts],
                                 dtype='int64'))

    # return
    return traces, case_dict, act_dict

# load alignment matrix
def load_amat(folname, filename):
    return np.loadtxt(folname+'/'+filename+'.out')

# get consensus sequence
def get_consensus(acts, freq, p):
    traces=acts.copy()
    freqs=freq.copy()
    freqs[freqs < p] = 0
    for i in range(len(acts)):
        if freqs[i] == 0:
            traces[i] = 0
    return traces

# get consensus activities
def get_cons_acts(cons, act_dict, freq):
    cons_acts_dict = {}
    cons_cols = []
    cons_acts = []
    cons_acts_num = []
    cons_col_freqs_dict = {}
    for index in range(len(cons)):
        if cons[index] > 0:
            cons_col_freqs_dict[index] = freq[index]
            cons_cols.append(index)
            cons_acts_num.append(cons[index])
            cons_acts_dict[cons[index]] = cons_acts_dict.get(cons[index], 0) + 1
            if cons_acts_dict[cons[index]] > 1:
                
                cons_acts.append(act_dict[cons[index]] + '_' + str(cons_acts_dict[cons[index]]))
            else:
                cons_acts.append(act_dict[cons[index]])
    return cons_cols, cons_acts, cons_acts_num, cons_acts_dict, cons_col_freqs_dict

# find the preceding consensus column
def find_last_cons_col(cons_cols, i):
    for idx in range(len(cons_cols) - 1):
        if cons_cols[idx + 1] > i:
            return cons_cols[idx]
    return 0

# find the succeeding consensus column
def find_next_cons_col(cons_cols, last_col, span):
    for idx in range(len(cons_cols)):
        if cons_cols[idx] == last_col:
            if idx + span <= len(cons_cols):
                return idx + span
            else:
                return cons_cols[len(cons_cols) - 1]
    return 0

# get the "backbone" edges (returns activity id)
def get_cons_edges(cons_acts):
    cs_edges = []
    for i in range(len(cons_acts) - 1):
        first = cons_acts[i]
        last = cons_acts[i + 1]
        cs_edges.append([first, last])
    return cs_edges

# get the "backbone" edges (returns the column index)
def get_cons_edges_col(cons_cols):
    cs_edges_col = []
    for i in range(len(cons_cols) - 1):
        first = cons_cols[i]
        last = cons_cols[i + 1]
        cs_edges_col.append([first, last])
    return cs_edges_col

# transfer column index to activity
def cols_to_acts(acts, act_dict, cs_edges_col, dis_edges_col, cons_col_freqs_dict, dis_col_freqs_dict):
    cols_acts_dict = {}
    acts_count_dict = {}
    cs_edges = []
    acts_count_dict[acts[cs_edges_col[0][0]]] = 1
    cols_acts_dict = {cs_edges_col[0][0]: act_dict[acts[cs_edges_col[0][0]]] }
    for edges in cs_edges_col:
        col = edges[1]
        frequency = cons_col_freqs_dict[col]
        act = acts[col]
        acts_count_dict[act] = acts_count_dict.get(act, 0) + 1
        if acts_count_dict[act] <= 1:
            cols_acts_dict[col] = act_dict[act]
        else:
            cols_acts_dict[col] = act_dict[act] + '_' + str(acts_count_dict[act])
        cs_edges.append([cols_acts_dict[edges[0]] + '(' + str(round(cons_col_freqs_dict[edges[0]], 2)) + ')', cols_acts_dict[col] + '(' + str(round(frequency, 2)) + ')'])

    col_freqs_dict = {}
    for key in cons_col_freqs_dict.keys():
        col_freqs_dict[key] = cons_col_freqs_dict[key]
    for key in dis_col_freqs_dict.keys():
        col_freqs_dict[key] = dis_col_freqs_dict[key]
    # dis col to act
    # dis_acts_count_dict = {}
    dis_edges = []
    if dis_edges_col:
        if dis_edges_col[0][0] not in cols_acts_dict:
            act = acts[dis_edges_col[0][0]]
            acts_count_dict[act] = 1
            cols_acts_dict[dis_edges_col[0][0]] = act_dict([act])

    for edges in dis_edges_col:
        col = edges[1]
        if col not in cols_acts_dict:
            act = acts[col]
            frequency = col_freqs_dict[col]
            acts_count_dict[act] = acts_count_dict.get(act, 0) + 1
            if acts_count_dict[act] <= 1:
                cols_acts_dict[col] = act_dict[act]
            else:
                cols_acts_dict[col] = act_dict[act] + '_' + str(acts_count_dict[act])
        dis_edges.append([cols_acts_dict[edges[0]] + '(' + str(round(col_freqs_dict[edges[0]], 2)) + ')', cols_acts_dict[col] + '(' + str(round(col_freqs_dict[col], 2)) + ')'])
    return cs_edges, dis_edges, col_freqs_dict

# construct the workflow model
def get_graph(fmt, fname, edges1, edges2):
    g = gv.Digraph(format=fmt)

    with g.subgraph(name='cluster_blue') as c:
        c.attr(style='filled')
        c.attr(color='white')
        c.node_attr.update(style='filled', color='lightblue2')
        for edge in edges1:
            c.edge(edge[0],edge[1])
    if edges2 != 0:
        g.attr('node', style='', color='black')
        for edge in edges2:
            # g.node(edge[0])
            # g.node(edge[1])
            g.edge(edge[0], edge[1])
    g.render(filename=fname)

# get common-but-dispersed activities
def get_dis_edges_and_plot(acts, cons_cols, cons_acts, cons_acts_num, cons, freq, dis_sum_threshold, act_dict, cons_edges):
    span = 30
    dis_edges_col = []
    dis_acts_dict = {}
    cons_acts_to_delete = []

    # for each activity number(label), append dis_edges
    for act in range(len(act_dict)):
        dis_act_start = 0 # start column of this dis act
        dis_act_end = 0 # end colunm of this dis act
        sum_of_freq = 0

    for i in range(len(cons_cols) - 1):
        dis_acts=[]
        dis_acts_num = []

        for j in range(cons_cols[i] + 1, cons_cols[i + 1]):
            left = cons_cols[i] + 1
            right = cons_cols[i + 1]
            idx = j
            indexes = [idxs for idxs in range(left, right) if acts[idxs] == acts[idx]] 

            if acts[idx] not in cons[left : right + 1] and acts[idx] not in dis_acts_num and np.sum(freq[indexes]) > dis_sum_threshold:
                dis_acts_dict[acts[idx]] = dis_acts_dict.get(acts[idx], 0) + 1
                if dis_acts_dict[acts[idx]] <= 1:
                    # identify duplicates to between dis_edges and cons_edges prevent reverse transition 
                    if acts[idx] in cons_acts_num:
                        dis_acts.append(act_dict[acts[idx]] + '_' + str(dis_acts_dict[acts[idx]]))
                    else:
                        dis_acts.append(act_dict[acts[idx]])
                # identify duplicates inside dis_edges
                else:
                    dis_acts.append(act_dict[acts[idx]] + '_' + str(dis_acts_dict[acts[idx]]))
                dis_acts_num.append(acts[idx])

        for k in range(len(dis_acts_num)):
            dis_edges.append([cons_acts[i], dis_acts[k]])
            dis_edges.append([dis_acts[k], cons_acts[i + 1]])
            # if has dis_acts between cons_acts, delete direct transition between cons_acts
            if cons_acts[i] not in cons_acts_to_delete:
                cons_acts_to_delete.append(cons_acts[i])
    return dis_edges, cons_acts_to_delete, dis_acts_dict

def model(fname, threshold, span):
    log = csv2log(filename=fname,
              caseids_col='caseID',
              acts_col='activity',
              starts_col='startTime')
    (traces, case_dict, act_dict) = log
    # Perform PIMA algorithm
    seed = 0
    # INITIALIZATION with random sequential method
    namat = ali.pima_init(traces, ('random','shuffle',seed)) 
    # alternate initialization with existing edit-distance methods
    # namat = pima_init(traces, ('tree','edit','single'))

    # SINGLE-TRACE ITERATIONS with random sequential method (1st convergence)
    while True:
        # perform iteration
        namat = ali.pima_iter(namat, ('tracewise','random',seed+len(namat)))
        # define convergence condition
        prev = namat[-2]['score']['sum_pair']
        next = namat[-1]['score']['sum_pair']
        if (prev - next) / prev <= 0.00:
            break

    # MULTI-TRACE ITERATION with range (0.1,0.9) by descending frequency
    namat = ali.pima_iter(namat, ('columnwise','range',(0.1,0.9),'des'))

    # SINGLE-TRACE ITERATIONS with random sequential method (2nd convergence)
    while True:
        # perform iteration
        namat = ali.pima_iter(namat, ('tracewise', 'random', seed+len(namat)))

        # define convergence condition
        prev = namat[-2]['score']['sum_pair']
        next = namat[-1]['score']['sum_pair']
        if (prev - next) / prev <= 0.00:
            break

    # RESULT
    amat = namat[-1]['amat'][:,:,0]
    # amat = load_amat(".", "113_emma")
    acts = np.max(amat, axis=0) # activity id
    freq = np.mean(amat != 0, axis=0) # the frequency of each column

    dis_edges_col = []
    dis_used_cols = []
    dis_sum_threshold = threshold
    # Add start and end
    numOfActs = len(act_dict)
    tLen = len(acts)
    acts = np.insert(acts, 0, numOfActs)
    acts = np.insert(acts, tLen + 1, numOfActs + 1)
    freq = np.insert(freq, 0, 1.0)
    freq = np.insert(freq, tLen + 1, 1.0)
    act_dict[numOfActs] = 'START'
    act_dict[numOfActs + 1] = 'END'
    cons = get_consensus(acts, freq, threshold)
    
    # get the consensus cols and corresponding acts
    cons_cols, cons_acts, cons_acts_num, cons_acts_dict, cons_col_freqs_dict = get_cons_acts(cons, act_dict, freq)
    if span == "max":
        span = len(cons_cols) - 1
    # add cons to edges
    cs_edges_col = get_cons_edges_col(cons_cols) # edges that uses col num to represent nodes
    # add distributed acts to edges
    # for each activity number(label), append dis_edges
    used_dis_col = []
    dis_col_freqs_dict = {}
    for act in range(len(act_dict)):
        for index in range(len(cons_cols) - span):
            dis_act_start = 0 # start column of this dis act
            dis_act_end = 0 # end colunm of this dis act
            cons_act_start = cons_cols[index]
            cons_act_end = cons_cols[index + span]
            sum_of_freq = 0

            # for each column, if act appears, add freq to sum
            i = cons_act_start
            # for i in range(cons_act_start, cons_act_end):
            while i < cons_act_end:
                if acts[i] == act:
                    
                    # not in cons_cols
                    if i in cons_cols:
                        i += 1
                        continue

                    # set dis_act_start
                    if sum_of_freq == 0: 
                        dis_act_start = i
                        
                    # set sum
                    sum_of_freq += freq[i]

                    # check if sum > threshold
                    if sum_of_freq >= dis_sum_threshold:
                        # set end
                        dis_act_end = i
                        # find last and next cons col
                        lastcol = 0
                        nextcol = 0
                        idx = 0
                        for j in range(len(cons_cols) - 1):
                            if cons_cols[j + 1] > dis_act_start:
                                lastcol = nextcol = cons_cols[j]
                                idx = j
                                break
                        for k in range(idx, len(cons_cols) - 1):
                            if cons_cols[k + 1] > dis_act_end:
                                nextcol = cons_cols[k + 1]
                                break
                        i += 1
                        while i < nextcol:
                            if acts[i] == act:
                                sum_of_freq += freq[i]
                                dis_act_end = i
                            i += 1
                        
                        dis_col_freqs_dict[dis_act_end] = sum_of_freq
                        sum_of_freq = 0

                        # add to edges
                        if [lastcol, dis_act_end] not in dis_edges_col and dis_act_end not in used_dis_col:
                            dis_edges_col.append([lastcol, dis_act_end])
                            used_dis_col.append(dis_act_end)
                        if [dis_act_end, nextcol] not in dis_edges_col:
                            dis_edges_col.append([dis_act_end, nextcol])
                        # reset
                        dis_act_start = i
                        dis_act_end = i
                i += 1

    cs_edges, dis_edges, col_freqs_dict = cols_to_acts(acts, act_dict, cs_edges_col, dis_edges_col, cons_col_freqs_dict, dis_col_freqs_dict)

    # create graph
    get_graph('pdf', 'img/TraumaIntub_' + str(threshold) + '_' + str(span), cs_edges, dis_edges)

def main():
    model("TraumaIntubationCoding.csv", 0.5, "max")

if __name__ == '__main__':
    main()  