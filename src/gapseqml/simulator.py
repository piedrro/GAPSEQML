import pandas as pd
import os
import json
import gapseqml
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pomegranate as pg
import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import tqdm
from tqdm import tqdm
import traceback


def calculate_cohens_d(trace, states):
    
    raw_data = np.array(trace)
    state_means = np.array(states)
    
    unique_states = np.unique(state_means)
    
    if len(unique_states) == 2:
    

        # Separate the traces by states
        state_0 = trace[states == unique_states[0]]
        state_1 = trace[states == unique_states[1]]
        
        mean_state_0 = np.mean(state_0)
        mean_state_1 = np.mean(state_1)
        
        std_state_0 = np.std(state_0, ddof=1)
        std_state_1 = np.std(state_1, ddof=1)
        
        # Calculate the sample sizes
        n_state_0 = len(state_0)
        n_state_1 = len(state_1)
        
        # Calculate the pooled standard deviation
        pooled_std = np.sqrt(((n_state_0 - 1) * std_state_0**2 + (n_state_1 - 1) * std_state_1**2) / (n_state_0 + n_state_1 - 2))
        
        # Calculate Cohen's d
        cohens_d = (mean_state_1 - mean_state_0) / pooled_std
    
    else:
        cohens_d = np.nan
    
    return abs(cohens_d)


def generate_n_state_means(min_state_dif, n_states):
    
    min_dif = 0
    
    if n_states > 1:
    
        while min_dif < min_state_dif:
        
            states = np.random.uniform(0.01, 0.99, n_states)
            diffs = np.diff(sorted(states))
            
            min_dif = min(diffs)

    else:
        
        states = np.random.uniform(0.01, 0.99)
        states = np.expand_dims(states, 0)
        
    states = np.sort(states)
        
    return states

def generate_transtion_matrix(n_states = 2, trans_prob=0.1):
    
    short_dwell = False
    
    if isinstance(trans_prob, list):
        trans_prob = np.random.uniform(min(trans_prob), max(trans_prob))
        
    matrix = np.empty([n_states, n_states])

    matrix.fill(trans_prob)
    np.fill_diagonal(matrix, 1 - trans_prob)

    if trans_prob != 0:
        stay_prob = 1 - trans_prob
        remaining_prob = 1 - matrix.sum(axis=0)
        matrix[matrix == stay_prob] += remaining_prob
    
    return matrix


def short_dwell_func(states, short_dwell_prob, dwell_range = [5,50]):
    
    try:
        
        short_dwell = False
        
        states = states.copy()
    
        if isinstance(short_dwell_prob, list):
            short_dwell_prob = np.random.randint(min(short_dwell_prob), max(short_dwell_prob)+1)
        
        n_states = len(np.unique(states))
        
        if n_states == 2:
            
            if np.random.uniform(0, 1) < short_dwell_prob:
                
                short_dwell = True
                
                is_one = (states == np.max(states))
                change_points = np.diff(is_one.astype(int), prepend=0, append=0)
                segments = np.where(change_points != 0)[0]
                subarrays = np.split(states, segments)
                subarrays = [arr for arr in subarrays if len(arr) > 0]
                
                for index, arr in enumerate(subarrays):
                    bind_length = np.random.randint(min(dwell_range),max(dwell_range))
                    if bind_length < len(arr):
                        arr[bind_length:] = min(states)
                        
                states = np.concatenate(subarrays)
            
    except:
        print(traceback.format_exc())
        pass
    
    return states
    

def compute_states(states, trace, trans_prob=None, bleached=None, bleach_index=None):
    
    state_means = np.unique(states).tolist()
    n_states = len(state_means)
    
    if n_states == 2:
        
        cohens_d = calculate_cohens_d(trace, states)
        
        differences = np.diff(states)
        n_transitions = np.count_nonzero(differences)
        state1_transitions = np.sum((states[:-1] == min(states)) & (states[1:] == max(states)))
        
        is_one = (states == np.max(states))
        change_points = np.diff(is_one.astype(int), prepend=0, append=0)
        segments = np.where(change_points != 0)[0]
        subarrays = np.split(states, segments)
        
        subarrays = [arr for arr in subarrays if len(arr) > 0]
        state1_lengths = [len(subarray) for subarray in subarrays if max(subarray) == max(states)]
        
        if len(state1_lengths) > 0:
            state1_max_length = max(state1_lengths)
        else:
            state1_max_length = 0

    else:
        cohens_d = 0
        n_transitions = 0
        state1_transitions = 0
        state1_max_length = 0
        
    stats = {"n_states": n_states,
             "state1_transitions": state1_transitions,
             "state1_max_length": state1_max_length,
             "state_means": state_means,
             "cohens_d": cohens_d,
             "bleached": bleached,
             "bleach_index":bleach_index,
             "trans_prob": trans_prob}
        
    return n_states, stats
    
    
def generate_nstate_trace(n_states = 2, trans_prob=[0.001,0.01], 
                          min_state_diff=0.3, eps = 1e-16, 
                          short_dwell_prob = 0.5, short_dwell_threshold = 50,
                          bleach_lifetime = None, noise = [0.01,0.1],
                          trace_length=1000):
    
    try:
    
        if isinstance(n_states, list):
            n_states = np.random.randint(min(n_states), max(n_states)+1)
                
        state_means = generate_n_state_means(min_state_diff, n_states)
        
        if type(state_means) == float:
            dists = [pg.NormalDistribution(state_means, eps)]
        else:
            dists = [pg.NormalDistribution(m, eps) for m in state_means]
            
        starts = np.random.uniform(0, 1, size=n_states)
        starts /= starts.sum()
        
        matrix = generate_transtion_matrix(n_states, trans_prob)
        
        model = pg.HiddenMarkovModel.from_matrix(
            matrix, distributions=dists, starts=starts)
        model.bake()
    
        final_matrix = model.dense_transition_matrix()[:n_states, :n_states]
    
        states = np.array(model.sample(n=1, length=trace_length))
        states = np.squeeze(states).round(4)
        
        states = short_dwell_func(states, short_dwell_prob)
        
        if isinstance(bleach_lifetime, list):
            bleach_lifetime = np.random.randint(min(bleach_lifetime), max(bleach_lifetime)+1)
            
        if n_states > 1 and isinstance(bleach_lifetime, int):
            bleach_index = int(np.ceil(np.random.exponential(bleach_lifetime)))
            states[bleach_index:] = np.min(states)
            bleached = True
        else:
            bleach_index = np.nan
            bleached = False
        
        trace = states.copy()
        noise_max = np.random.uniform(min(noise), max(noise))
        trace = trace + np.random.normal(0, noise_max, len(trace))
        
        n_states, stats = compute_states(
            states, trace, trans_prob, bleached, bleach_index)
        
        if n_states == 2:
            if stats["state1_max_length"] > short_dwell_threshold:
                label = 2
            else:
                label = 1
        else:
            label = 0
            
        stats["label"] = label

    except:
        print(traceback.format_exc())
        trace, states, stats = None, None, None
        
    return trace, states, stats
    
# trace, states, trace_info = generate_nstate_trace()

# plt.plot(trace)
# plt.plot(states)
# plt.title(f"max_length:{trace_info['state1_max_length']}, n_transitions:{trace_info['state1_transitions']}")
# plt.show()


if __name__ == '__main__':
    
    simulated_traces = []
    simulated_labels = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(generate_nstate_trace) for i in range(50000)]
    
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                trace, states, trace_info = future.result()
                
                if trace is not None:
                    simulated_traces.append(trace)
                    label = trace_info["label"]
                    simulated_labels.append(label)

            except:
                pass
            
    unique_labels, label_counts = np.unique(simulated_labels, return_counts=True)
    
    simulated_traces = np.array(simulated_traces)
    simulated_labels = np.array(simulated_labels)
    
    permutation = np.random.permutation(len(simulated_traces))
    simulated_traces = simulated_traces[permutation]
    simulated_labels = simulated_labels[permutation]
    
    label_max = 1000
    
    index_selection = []
    
    for label in unique_labels:
        
        label_indexes = np.argwhere(simulated_labels==label)[:,0]
        label_indexes = label_indexes[:label_max]
        
        index_selection.extend(label_indexes)
          
    simulated_traces = np.take(simulated_traces, index_selection,axis=0)
    simulated_labels = np.take(simulated_labels, index_selection, axis=0)
    
    permutation = np.random.permutation(len(simulated_traces))
    simulated_traces = simulated_traces[permutation]
    simulated_labels = simulated_labels[permutation]
    
    simulated_labels = [label.tolist() for label in simulated_labels]
    simulated_traces = [trace.tolist() for trace in simulated_traces]
            
    json_dict = {}
    json_dict["simulated_data"] = simulated_traces
    json_dict["label"] = simulated_labels
    
    gapseqml_dir = os.path.dirname(gapseqml.__file__)
    gapseqml_dir = os.path.dirname(gapseqml_dir)
    gapseqml_dir = os.path.dirname(gapseqml_dir)
        
    json_dir = os.path.join(gapseqml_dir, "data", "train", "simulated")

    if not os.path.exists(json_dir):
        os.mkdir(json_dir)
            
    json_path = os.path.join(json_dir, "simulated_data.txt")
    
    with open(json_path, "w") as f:
        json.dump(json_dict, f)
            
            
            
            
            
            
            
            
            
            
            