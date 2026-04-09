import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
import networkx as nx
def is_present(value):
    return value != "0"

def generate_heatmap(data_matrix, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 10))
    # Make symmetric by taking upper triangle and mirroring it
    i_lower = np.tril_indices_from(data_matrix, -1)
    data_matrix[i_lower] = data_matrix.T[i_lower]
    np.fill_diagonal(data_matrix, 0)
    ax = sns.heatmap(
    data_matrix,
    cmap="coolwarm",
    cbar=True,
    center=0
    )
    step = 20
    tick = list(range(0, data_matrix.shape[0], step))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.xticks(tick, tick)
    plt.yticks(tick, tick)
    plt.title(title)
    plt.xlabel(xlabel, rotation=0)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

# Need to change this function to only print the clusters instead of all NPs
def generate_cluster_heatmap_full(cluster_np_indices, window_np_presence, hist1_window_indices, title, filename):

    if len(cluster_np_indices) == 0:
        print("Cluster is empty. Skipping heatmap.")
        return

    # Sort cluster indices for cleaner visualization
    cluster_np_indices_sorted = sorted(cluster_np_indices)

    # Build matrix: cluster NPs × Hist1 windows
    cluster_matrix = np.array([
        [1 if window_np_presence[w][np_idx] else 0 for w in hist1_window_indices]
        for np_idx in cluster_np_indices_sorted
    ], dtype=int)

    # Create labels
    row_labels = [f"NP{idx+1}" for idx in cluster_np_indices_sorted]
    col_labels = [f"W{i}" for i in range(len(hist1_window_indices))]

    df = pd.DataFrame(cluster_matrix, index=row_labels, columns=col_labels)

    # Plot heatmap
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(df, cmap="Greys", cbar=True, vmin=0, vmax=1)

    ax.set_title(title)
    ax.set_xlabel("Hist1 windows")
    ax.set_ylabel("Cluster NPs")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def k_means_clustering(cluster1, cluster2, cluster3, jaccard_distance_normalized_matrix, initial_centers):
    new_centers = []

    clusters = [cluster1, cluster2, cluster3]

    for i, cluster in enumerate(clusters):
        if len(cluster) == 0:
            #print(f"Cluster {i+1} is empty. Keeping previous center:", initial_centers[i])
            new_centers.append(initial_centers[i])
            continue

        total_distances = []
        for cluster_np in cluster:
            total = sum(
                jaccard_distance_normalized_matrix[cluster_np][other_np]
                for other_np in cluster
            )
            total_distances.append(total)

        minimum_distance = np.argmin(total_distances)
        new_centers.append(cluster[minimum_distance])

        #print(f"Cluster {i+1} new center:", cluster[minimum_distance])

    #print("\nNew cluster centers after iteration:", new_centers)
    return new_centers

# Says it uses jaaccard_normalized but it uses then normal jaccard (Need to change the function variable name to reflect this)
def k_medoids(all_indices, jaccard_distance_matrix, initial_centers):
    iteration = 0
    seen_centers = set()
    while(True):
        # Assign previous centers to compare to new centers after reassignment
        previous_centers = initial_centers.copy()
        previous_centers_tuple = tuple(previous_centers)

        # Make a list to store cluster assignments for each NP, initialized to 0
        cluster_assignments = np.zeros(len(all_indices), dtype=int)

        # Calculate distance and assign to closest center for each NP
        for idx, np_idx in enumerate(all_indices):
            distances = [
                jaccard_distance_matrix[np_idx][center] for center in initial_centers
            ]
            cluster_assignments[idx] = np.argmin(distances)
            
        # Looking through all indices and assigning their cluster to an array
        cluster1 = [all_indices[i] for i in range(len(all_indices)) if cluster_assignments[i] == 0]
        cluster2 = [all_indices[i] for i in range(len(all_indices)) if cluster_assignments[i] == 1]
        cluster3 = [all_indices[i] for i in range(len(all_indices)) if cluster_assignments[i] == 2]

        # Reassign centers 
        initial_centers = k_means_clustering(cluster1, cluster2, cluster3, jaccard_distance_matrix, initial_centers)
            
        # Check if they are the same 
        if previous_centers == initial_centers:
            #print("\nK-medoids clustering converged.")
            break

        # Check if the center has been seen before
        if previous_centers_tuple in seen_centers:
            #print("\nK-medoids clustering entered a loop. Stopping.")
            break

        # Add the previous centers to the seen set
        seen_centers.add(previous_centers_tuple)
        iteration += 1
            
        # Increment counter 
        if iteration > 1000:
            #print("\nK-medoids clustering reached maximum iterations.")
            break
        
        # Print final cluster centers and their NP members
        #print("\n=== FINAL CLUSTERING RESULTS ===")
        #print("\nFinal cluster centers:", initial_centers)
        #print("\nFinal Cluster 1 NPs:", cluster1)
        #print("\nFinal Cluster 2 NPs:", cluster2)
        #print("\nFinal Cluster 3 NPs:", cluster3)
    # Variation is the sum of the distance of each NP to its assigned center
    variation1 = sum(jaccard_distance_matrix[np_idx][initial_centers[0]] for np_idx in cluster1)
    variation2 = sum(jaccard_distance_matrix[np_idx][initial_centers[1]] for np_idx in cluster2)
    variation3 = sum(jaccard_distance_matrix[np_idx][initial_centers[2]] for np_idx in cluster3)
    
    total_variation = (variation1 + variation2 + variation3) 
    return initial_centers, cluster1, cluster2, cluster3, total_variation

        #print("\nCluster 1 variation:", variation1)
        #print("Cluster 2 variation:", variation2)
        #print("Cluster 3 variation:", variation3)
        #print("Total within-cluster variation:", total_variation)

def compute_feature_percentages(cluster, window_np_presence, hist1_windows_indices, hist1_features, lad_features, 
    vmn_features, rnapii_s2p, rnapii_s5p, rnapii_s7p, enhancer, h3k9me3, h3k20me3, h3k27me3, h3k36me3, nanog, pou5f1, sox2, ctcf):                
    # Initialize percentages 
    hist1_percentages = []
    lad_percentages = []
    vmn_percentages = []
    rnapii_s2p_percentages = [] 
    rnapii_s5p_percentages = []
    rnapii_s7p_percentages = []
    enhancer_percentages = []
    h3k9me3_percentages = []
    h3k20me3_percentages = []
    h3k27me3_percentages = []
    h3k36me3_percentages = []
    nanog_percentages = []
    pou5f1_percentages = []
    sox2_percentages = []
    ctcf_percentages = []

    # Num NPs in Cluster
    for np_idx in cluster:
        # Initialize 
        total_detected = 0
        hist1_count = 0
        lad_count = 0
        vmn_count = 0
        rnapii_s2p_count = 0
        rnapii_s5p_count = 0
        rnapii_s7p_count = 0
        enhancer_count = 0
        h3k9me3_count = 0
        h3k20me3_count = 0
        h3k27me3_count = 0
        h3k36me3_count = 0
        nanog_count = 0
        pou5f1_count = 0
        sox2_count = 0
        ctcf_count = 0
        
        # iterate only over Hist1 windows, Local I to avoid issues 
        # Local I is like 0 - number of hist1 windows
        # w_idx is global list and in the tens of thousands, but we want to compare to the hist1_region.csv which is smaller
        for local_i, w_idx in enumerate(hist1_windows_indices):
            # Check if this NP detected this window, if so increment total detected and then check for features
            if window_np_presence[w_idx][np_idx]:
                # Count number of windows detected in Hist1 region for this NP, and how many of those windows have the feature
                total_detected += 1
                # If the feature is present in this window, increment the count for that feature for this NP
                if hist1_features[local_i] == 1:
                    hist1_count += 1
                # If present in this window, increment LAD counter 
                if lad_features[local_i] == 1:
                    lad_count += 1
                # For the last project, continue this for each feature we want to compute percentages for 
                if vmn_features[local_i] == 1:
                    vmn_count += 1
                if rnapii_s2p[local_i] == 1:
                    rnapii_s2p_count += 1
                if rnapii_s5p[local_i] == 1:
                    rnapii_s5p_count += 1
                if rnapii_s7p[local_i] == 1:
                    rnapii_s7p_count += 1
                if enhancer[local_i] == 1:
                    enhancer_count += 1
                if h3k9me3[local_i] == 1:
                    h3k9me3_count += 1
                if h3k20me3[local_i] == 1:
                    h3k20me3_count += 1
                if h3k27me3[local_i] == 1:
                    h3k27me3_count += 1
                if h3k36me3[local_i] == 1:
                    h3k36me3_count += 1
                if nanog[local_i] == 1:
                    nanog_count += 1
                if pou5f1[local_i] == 1:
                    pou5f1_count += 1
                if sox2[local_i] == 1:
                    sox2_count += 1
                if ctcf[local_i] == 1:
                    ctcf_count += 1

        # Calculate percentages and append to a list to average later 
        if total_detected > 0:
            hist1_percentages.append(hist1_count / total_detected)
            lad_percentages.append(lad_count / total_detected)
            vmn_percentages.append(vmn_count / total_detected)
            rnapii_s2p_percentages.append(rnapii_s2p_count / total_detected)
            rnapii_s5p_percentages.append(rnapii_s5p_count / total_detected)
            rnapii_s7p_percentages.append(rnapii_s7p_count / total_detected)
            enhancer_percentages.append(enhancer_count / total_detected)
            h3k9me3_percentages.append(h3k9me3_count / total_detected)
            h3k20me3_percentages.append(h3k20me3_count / total_detected)
            h3k27me3_percentages.append(h3k27me3_count / total_detected)
            h3k36me3_percentages.append(h3k36me3_count / total_detected)
            nanog_percentages.append(nanog_count / total_detected)
            pou5f1_percentages.append(pou5f1_count / total_detected)
            sox2_percentages.append(sox2_count / total_detected)
            ctcf_percentages.append(ctcf_count / total_detected)
        # Else: Append 0 to avoid dividing by 0 and to reflect that if no windows were detected, then 0% of detected windows had the feature
        else:
            hist1_percentages.append(0)
            lad_percentages.append(0)
            vmn_percentages.append(0)
            rnapii_s2p_percentages.append(0)
            rnapii_s5p_percentages.append(0)
            rnapii_s7p_percentages.append(0)
            enhancer_percentages.append(0)
            h3k9me3_percentages.append(0)
            h3k20me3_percentages.append(0)
            h3k27me3_percentages.append(0)
            h3k36me3_percentages.append(0)
            nanog_percentages.append(0)
            pou5f1_percentages.append(0)
            sox2_percentages.append(0)
            ctcf_percentages.append(0)

    return hist1_percentages, lad_percentages, vmn_percentages, rnapii_s2p_percentages, rnapii_s5p_percentages, rnapii_s7p_percentages, enhancer_percentages, h3k9me3_percentages, h3k20me3_percentages, h3k27me3_percentages, h3k36me3_percentages, nanog_percentages, pou5f1_percentages, sox2_percentages, ctcf_percentages

def main():
    filename = "GSE64881_segmentation_at_30000bp.passqc.multibam (2).txt"
    filename2 = "Hist1_region_features.csv"

    with open(filename, "r") as f:

        # Store coords in list for later use
        window_coords = []
        #header
        window_np_presence = [] 
        # Get first line
        header = f.readline().strip()
        # Split header into columns
        columns = header.split()

        # Takes everything after chrom/start/stop as NP headers
        np_headers = columns[3:]

        # Count NP columns (those whose header contains "F")
        num_nps = 0
        for h in np_headers:
            if "F" in h:
                num_nps += 1
        # Initialize variables for stats
        num_windows = 0
        windows_per_np = [0] * num_nps
        total_nps_per_window = 0
        min_nps_per_window = None
        max_nps_per_window = 0

        # Window frequency data for compaction estimate
        window_detection = []

        #Now read the *data* lines
        for line in f:
            # Break the line up
            line = line.strip()
            if not line:
                continue
            # Split the line into parts and check if it has enough columns (chrom, start, stop + NP values)
            parts = line.split()
            if len(parts) < 3 + num_nps:
                # skip malformed line
                continue
            
            # First column
            chrom = parts[0]
            # Second column
            start = int(parts[1])
            # Third column
            stop = int(parts[2])
            # Store the window coordinates for later use
            window_coords.append((chrom, start, stop))

            num_windows += 1

            # everything after chrom/start/stop
            np_values = parts[3:]
            nps_this_window = 0
            presence = []

            for idx in range(num_nps):
                val = np_values[idx]
                # Check for a 1 or 0 and store presence for this NP in this window
                detected = is_present(val)
                presence.append(detected) # True if detected, False if not
                # If detected, increment the count of windows for this NP and the count of NPs for this window
                if detected:
                    windows_per_np[idx] += 1
                    nps_this_window += 1
    
            window_np_presence.append(presence) # List of booleans indicating presence of each NP in this window    
            
            total_nps_per_window += nps_this_window
            # Assign min and max NPs per window for this window
            if min_nps_per_window is None or nps_this_window < min_nps_per_window:
                min_nps_per_window = nps_this_window
            if nps_this_window > max_nps_per_window:
                max_nps_per_window = nps_this_window

            # Store detection frequency for this window
            window_detection.append(nps_this_window / num_nps)

        # Calculate detection frequency per NP (Windows per NP / Total Windows)
        detection_frequencies = [w / num_windows for w in windows_per_np]
        
        # Calculate mean and solve for good threshhold for outliers
        mean_freq = np.mean(detection_frequencies)
        std_freq = np.std(detection_frequencies)
        outlier_threshold = mean_freq + 2 * std_freq
        
        # Assignment 2 - radial position and compaction estimates
        # Calculate radial positions based on percentiles
        # Store frequencies in an array for easier percentile calculations
        freqs = np.array(detection_frequencies)
        #The percentile finds the cutoffs for each radial position, IE: at what number is 20% of NPs below q20 and 80% above it
        # Low frequency = apical = low radial position
        q20, q40, q60, q80 = np.percentile(freqs, [20, 40, 60, 80])

        radial_position = []
        for fval in freqs:
            if fval <= q20:
                pos = 1
            elif fval <= q40:
                pos = 2
            elif fval <= q60:
                pos = 3
            elif fval <= q80:
                pos = 4
            else:
                pos = 5
            radial_position.append(pos)
            
        # Output outliers based on detection frequency
        for idx, freq in enumerate(detection_frequencies):
            outlier_tag = "OUTLIER HIGH" if freq > outlier_threshold else ""
            #print only outliers
            if outlier_tag:
                print(f"NP {idx+1}: Detection Frequency = {freq:.4f} {outlier_tag}")
            
        window_array = np.array(window_detection)
        # Start finding the estimated compaction based on frequency (Basically inverse of radial position with more bins)
        # Lower frequency = higher compaction
        # Opposite of radial position, but with 10 bins instead of 5
        l10,l20,l30,l40,l50,l60,l70,l80,l90 = np.percentile(window_array, [10,20,30,40,50,60,70,80,90])

        estimated_compaction = []
        for fval in window_detection: 
            if fval <= l10:
                compaction = 10
            elif fval <= l20:
                compaction = 9 
            elif fval <= l30:
                compaction = 8
            elif fval <= l40:
                compaction = 7
            elif fval <= l50:
                compaction = 6
            elif fval <= l60:
                compaction = 5
            elif fval <= l70:
                compaction = 4
            elif fval <= l80:
                compaction = 3
            elif fval <= l90:
                compaction = 2 
            else: 
                compaction = 1
            estimated_compaction.append(compaction)

        # General Info
        print("\n=== DATASET SUMMARY ===")
        print("Number of genomic windows:", num_windows)
        print("Number of NPs:", num_nps)
        # Windows
        print("\n=== WINDOWS PER NP ===")
        print("Average windows per NP:", np.mean(windows_per_np))
        print("Minimum windows in any NP:", min(windows_per_np))
        print("Maximum windows in any NP:", max(windows_per_np))
        # NPS
        print("\n=== NPs PER WINDOW ===")
        print("Average NPs per window:", total_nps_per_window / num_windows)
        print("Minimum NPs per window:", min_nps_per_window)
        print("Maximum NPs per window:", max_nps_per_window)
        # Outlier information 
        print("\n=== OUTLIER THRESHOLD ===")
        print("Mean detection frequency:", mean_freq)
        print("Std detection frequency:", std_freq)
        print("Outlier threshold (mean + 2*std):", outlier_threshold)
        # Number of each radial position
        print("\n=== RADIAL POSITION DISTRIBUTION ===")
        print("Radial position 1 (strongly apical):", radial_position.count(1))
        print("Radial position 2:", radial_position.count(2))
        print("Radial position 3:", radial_position.count(3))
        print("Radial position 4:", radial_position.count(4))
        print("Radial position 5 (strongly equatorial):", radial_position.count(5))
        # Number of each compaction 
        print("\n=== COMPACTION SUMMARY ===")
        for c in range(1, 11):
            print(f"Compaction {c}:", estimated_compaction.count(c))

        # Start Assignment 3 
        # Hist1 located chromosome 13, Start 21.7 Mb, Stop 24.1 Mb
        # Extract Hist1 region data
        hist1_chrom = "chr13"
        hist1_start = 21700000
        hist1_stop = 24100000
        # Declare Variables for Hist1 analysis
        hist1_windows_indices = []
        hist1_np_indices = set() 
        windows_per_np_hist1 = [0] * num_nps
        nps_per_windows_hist1 = []
        # Analyze each window to see if it overlaps with Hist1 region
        for w_idx, (chrom, start, stop) in enumerate(window_coords):
            # Check if its between start and stop and is right chromosome
            if chrom == hist1_chrom and not (stop < hist1_start or start > hist1_stop):
                # Count windows overlapping Hist1 region
                hist1_windows_indices.append(w_idx)
                # Count Nps detecting Hist1 region / reset counter for the window
                nps_detecting = 0
                # Go through ALL NPs and Check if they detected this window, if so add to the count for this window and add to the set of NPs that detected Hist1
                for np_idx in range(num_nps):
                    # Check if NP detected this window, w_idx = local, np_idx = global, but window_np_presence is global so we can check directly
                    if window_np_presence[w_idx][np_idx]:
                        # Currently detected 
                        nps_detecting += 1
                        hist1_np_indices.add(np_idx)
                        windows_per_np_hist1[np_idx] += 1
                nps_per_windows_hist1.append(nps_detecting)

        print ("\n=== HIST1 REGION ANALYSIS ===")
        print("Numer of genomic windows: ", len(hist1_windows_indices))
        print("Number of Nps detecting Hist1:", len(hist1_np_indices))
        # Windows per NP
        print("Windows per NP:")
        # Only consider NPs that detected at least 1 window for these stats
        nonzero_windows = [w for w in windows_per_np_hist1 if w > 0]
        print("Average windows per NP:", np.mean(nonzero_windows))
        print("Minimum windows in any NP:", min(nonzero_windows))
        print("Maximum windows in any NP:", max(nonzero_windows))
        # Nps per window
        print("NPs per window:")
        print("Average NPs per window:", np.mean(nps_per_windows_hist1))
        print("Minimum NPs per window:", min(nps_per_windows_hist1))
        print("Maximum NPs per window:", max(nps_per_windows_hist1))
        # Radial position summary 
        print("\n=== HIST1 RADIAL POSITION DISTRIBUTION ===")
        # Assign radial postiion based by comparing to global radial position list
        hist_radial_position = [radial_position[i] for i in hist1_np_indices]
        for r in range(1, 6):
            print(f"Radial position {r}:", hist_radial_position.count(r))
        # Compaction Summary, assigned by comparing to global compaction list
        hist1_compaction = [estimated_compaction[i] for i in hist1_windows_indices]
        print("\n=== HIST1 COMPACTION SUMMARY ===")
        for c in range(1, 11):
            print(f"Compaction {c}:", hist1_compaction.count(c))

        # Assignment 4/5
        # Computer Jaccard Index for Relevant NPs
        # Store in Arrays that are the size of num_nps
        jaccard_index_matrix = np.zeros((num_nps, num_nps))
        jaccard_distance_matrix = np.zeros((num_nps, num_nps))        
        # Store hist1 NPs in a list 
        hist1_np_indices = list(hist1_np_indices)
        # Compare each NP to every other NP
        for i in range(len(hist1_np_indices)):
            for j in range(i + 1, len(hist1_np_indices)):
                # NP and NP + 1 
                np1 = hist1_np_indices[i]
                np2 = hist1_np_indices[j]
                # Get windows detected by each NP
                windows_np1 = set()
                windows_np2 = set() 
                # Check through each window in Hist1 region
                for w_idx in hist1_windows_indices:
                    if window_np_presence[w_idx][np1]:
                        windows_np1.add(w_idx)
                    if window_np_presence[w_idx][np2]:
                        windows_np2.add(w_idx)
                # Intersection Checks if they're the same
                intersection = len(windows_np1.intersection(windows_np2))
                # Union checks for either 
                union = len(windows_np1.union(windows_np2))
                # Divide the intersection by their union to get Jaccard Index
                jaccard_index = intersection / union if union > 0 else 0
                #print(f"Jaccard Index between NP {np1+1} and NP {np2+1}: {jaccard_index:.4f}")
                jaccard_index_matrix[np1][np2] = jaccard_index
                # Start Jaccard Distance 
                jaccard_distance = 1 - jaccard_index
                #print(f"Jaccard Distance between NP {np1+1} and NP {np2+1}: {jaccard_distance:.4f}")
                jaccard_distance_matrix[np1][np2] = jaccard_distance
                jaccard_distance_matrix[np2][np1] = jaccard_distance
        
        # Convert hist1_np_indices to a sorted list for matrix 
        hist1_np_indices_sorted = sorted(hist1_np_indices)  # or sort by windows detected for trend

        # --- Similarity Heatmap ---
        sim_subset = jaccard_index_matrix[np.ix_(hist1_np_indices_sorted, hist1_np_indices_sorted)].copy()
        generate_heatmap(
            sim_subset,
            "Jaccard Similarity Heatmap for HIST1 NPs (Function)",
            "NP Index",
            "NP Index",
            "jaccard_similarity_heatmap_hist1_test_function.png"
        )
        # --- Distance Heatmap ---
        dist_subset = jaccard_distance_matrix[np.ix_(hist1_np_indices_sorted, hist1_np_indices_sorted)].copy()
        generate_heatmap(
            dist_subset,
            "Jaccard Distance Heatmap for HIST1 NPs (Function)",
            "NP Index",
            "NP Index",
            "jaccard_distance_heatmap_hist1_test_function.png"
        )

        # Assignment 6/7: Clustering
        # Normalize our matrices
        random_seeds = np.random.choice(hist1_np_indices, size=3, replace=False)
        # New arrays for normalized values
        jaccard_similarity_normalized_matrix = np.zeros((num_nps, num_nps))
        jaccard_distance_normalized_matrix = np.zeros((num_nps, num_nps))
        for i in range(len(hist1_np_indices)):
            for j in range(i + 1, len(hist1_np_indices)):
                np1 = hist1_np_indices[i]
                np2 = hist1_np_indices[j]
                windows_np1 = set()
                windows_np2 = set()
                # Normalize by mboth / min(a, b)
                for w_idx in hist1_windows_indices:
                    # Sum np1 and np2 presence
                    if window_np_presence[w_idx][np1]:
                        windows_np1.add(w_idx)
                    if window_np_presence[w_idx][np2]:
                        windows_np2.add(w_idx)
                # Calculate intersection and unions
                intersection = len(windows_np1.intersection(windows_np2))   
                A_union = len(windows_np1)
           
                B_union = len(windows_np2)
                ab_union = min(A_union, B_union)
                # Jaccard similarity normalized
                if ab_union > 0:
                    jaccard_similarity_normalized = intersection/ab_union
                else:
                    jaccard_similarity_normalized = 0
                jaccard_similarity_normalized_matrix[np1][np2] = jaccard_similarity_normalized
                jaccard_similarity_normalized_matrix[np2][np1] = jaccard_similarity_normalized
               
                # Jaccard distance normalized 
                jaccard_distance_normalized = 1 - jaccard_similarity_normalized
                jaccard_distance_normalized_matrix[np1][np2] = jaccard_distance_normalized
                jaccard_distance_normalized_matrix[np2][np1] = jaccard_distance_normalized
              
        # Heatmap for normalized similarity matrix
        sim_norm_subset = jaccard_similarity_normalized_matrix[np.ix_(hist1_np_indices_sorted, hist1_np_indices_sorted)].copy()
        generate_heatmap(
            sim_norm_subset,
            "Normalized Jaccard Similarity Heatmap for HIST1 NPs (Function)",
            "NP Index",
            "NP Index",
            "jaccard_similarity_normalized_heatmap_hist1_test_function.png"
        )
        # Heatmap for normalized distance matrix
        dist_norm_subset = jaccard_distance_normalized_matrix[np.ix_(hist1_np_indices_sorted, hist1_np_indices_sorted)].copy()
        generate_heatmap(
            dist_norm_subset,
            "Normalized Jaccard Distance Heatmap for HIST1 NPs (Function)",
            "NP Index",
            "NP Index",
            "jaccard_distance_normalized_heatmap_hist1_test_function.png"
        )

        # K-means clustering with k=3, random initialization
        k = 3
        all_indices = list(hist1_np_indices_sorted)
        # Randomly select 3 intial centers from hist1_np_indices (163) 
        initial_centers = random.sample(all_indices, k)
        # Make a list to store cluster assignments for each NP, initialized to 0
        cluster_assignments = np.zeros(len(all_indices), dtype=int)

        for i, np_idx in enumerate(all_indices):
            distances = [
                # Use normalized distance to assign clusters (distance from center in that matrix)
                jaccard_distance_matrix[np_idx][center] for center in initial_centers
            ]
            # Assign to the cluster with the minimum distance 
            cluster_assignments[i] = np.argmin(distances)

        #print("\n=== K-MEANS CLUSTERING RESULTS ===")
        #print("\Initial cluster centers:", initial_centers)
        #print("Num clusters assigned:", np.bincount(cluster_assignments))
        # Looking through all indices and assigning their cluster to an array 
        cluster1 = [all_indices[i] for i in range(len(all_indices)) if cluster_assignments[i] == 0]
        cluster2 = [all_indices[i] for i in range(len(all_indices)) if cluster_assignments[i] == 1]
        cluster3 = [all_indices[i] for i in range(len(all_indices)) if cluster_assignments[i] == 2]
        #print("\nCluster 1 initial value:", initial_centers[0], "Number of cluster 1 NPs:", len(cluster1), "Cluster 1 NPs:", cluster1)
        #print("\nCluster 2 initial value:", initial_centers[1], "Number of cluster 2 NPs:", len(cluster2), "Cluster 2 NPs:", cluster2)
        #print("\nCluster 3 initial value:", initial_centers[2], "Number of cluster 3 NPs:", len(cluster3), "Cluster 3 NPs:", cluster3)
        print("\nOriginal cluster lengths:", len(cluster1), len(cluster2), len(cluster3))
        # FEATURE SELECTION ACTIVITY 1 
        best_variation = float("inf")
        best_result = None
        # Run k-medoids multiple times and keep the best result
        for run in range(1000): 
            # New random centers for each run to find the best clustering result
            initial_centers = random.sample(all_indices, k)
            
            centers, cluster1, cluster2, cluster3, variation = k_medoids(
                all_indices,
                jaccard_distance_matrix,
                initial_centers
            )

            #print(f"\nRun {run+1}")
            #print("Centers:", centers)
            #print("Variation:", variation)

            # Keep track of the best variation and corresponding centers and clusters
            if variation < best_variation:
                best_variation = variation
                best_result = (centers, cluster1, cluster2, cluster3)

        print("\n=== BEST OVER ALL RUNS ===")
        print("Best variation:", best_variation)
        print("Best centers:", best_result[0])
        # Create a heatmap for the best set of clusters, Rows = NPs, Columns = Centers, Cells = values from seg table (0 or 1)
        # Create a matrix to represent the heatmap data
        generate_cluster_heatmap_full(
            best_result[1],
            window_np_presence,
            hist1_windows_indices,
            "Cluster Heatmap for cluster 1 NPs (Best K-Medoids Result)",
            "cluster_heatmap_1.png"
        )

        generate_cluster_heatmap_full(
            best_result[2],
            window_np_presence,
            hist1_windows_indices,
            "Cluster Heatmap for cluster 2 NPs (Best K-Medoids Result)",
            "cluster_heatmap_2.png"
        )

        generate_cluster_heatmap_full(
            best_result[3],
            window_np_presence,
            hist1_windows_indices,
            "Cluster Heatmap for cluster 3 NPs (Best K-Medoids Result)",
            "cluster_heatmap_3.png"
        )

        print("\nCluster 1 NPs:", best_result[1])
        print("Cluster 2 NPs:", best_result[2])
        print("Cluster 3 NPs:", best_result[3])

        print("\n Final cluster lengths", len(best_result[1]), len(best_result[2]), len(best_result[3]))
        
        # FEATURE SELECTION ACTIVITY 2 
        df = pd.read_csv("Hist1_region_features.csv")
        # Extract Hist1 and Lad from file 
        hist1_features = df["Hist1"].astype(int).tolist()
        lad_features = df["LAD"].astype(int).tolist()
        vmn_features = df["Vmn"].astype(int).tolist()
        rnapii_s2p = df["RNAPII-S2P"].astype(int).tolist()
        rnapii_s5p = df["RNAPII-S5P"].astype(int).tolist()
        rnapii_s7p = df["RNAPII-S7P"].astype(int).tolist()
        enhancer = df["Enhancer"].astype(int).tolist()
        h3k9me3 = df["H3K9me3"].astype(int).tolist()
        h3k20me3 = df["H3K20me3"].astype(int).tolist()
        h3k27me3 = df["h3k27me3"].astype(int).tolist()
        h3k36me3 = df["H3K36me3"].astype(int).tolist()
        nanog = df["NANOG"].astype(int).tolist()
        pou5f1 = df["pou5f1"].astype(int).tolist()
        sox2 = df["sox2"].astype(int).tolist()
        ctcf = df["CTCF-7BWU"].astype(int).tolist()

        # Call function to compute percentages for each cluster and print results
        (cluster1_hist1, cluster1_lad, cluster1_vmn, cluster1_rnapii_s2p, cluster1_rnapii_s5p, cluster1_rnapii_s7p, cluster1_enhancer, 
        cluster1_h3k9me3, cluster1_h3k20me3, cluster1_h3k27me3, cluster1_h3k36me3, cluster1_nanog, cluster1_pou5f1, cluster1_sox2, cluster1_ctcf) = compute_feature_percentages(
            best_result[1],
            window_np_presence,
            hist1_windows_indices,
            hist1_features,
            lad_features,
            vmn_features,
            rnapii_s2p,
            rnapii_s5p,
            rnapii_s7p,
            enhancer,
            h3k9me3,
            h3k20me3,
            h3k27me3,
            h3k36me3,
            nanog,
            pou5f1,
            sox2,
            ctcf
        )

        # Calculate average percentage of feature
        cluster1_hist1_avg = np.mean(cluster1_hist1) if cluster1_hist1 else 0
        cluster1_lad_avg = np.mean(cluster1_lad) if cluster1_lad else 0
        cluster1_vmn_avg = np.mean(cluster1_vmn) if cluster1_vmn else 0
        cluster1_rnapii_s2p_avg = np.mean(cluster1_rnapii_s2p) if cluster1_rnapii_s2p else 0
        cluster1_rnapii_s5p_avg = np.mean(cluster1_rnapii_s5p) if cluster1_rnapii_s5p else 0
        cluster1_rnapii_s7p_avg = np.mean(cluster1_rnapii_s7p) if cluster1_rnapii_s7p else 0
        cluster1_enhancer_avg = np.mean(cluster1_enhancer) if cluster1_enhancer else 0  
        cluster1_h3k9me3_avg = np.mean(cluster1_h3k9me3) if cluster1_h3k9me3 else 0
        cluster1_h3k20me3_avg = np.mean(cluster1_h3k20me3) if cluster1_h3k20me3 else 0  
        cluster1_h3k27me3_avg = np.mean(cluster1_h3k27me3) if cluster1_h3k27me3 else 0
        cluster1_h3k36me3_avg = np.mean(cluster1_h3k36me3) if cluster1_h3k36me3 else 0
        cluster1_nanog_avg = np.mean(cluster1_nanog) if cluster1_nanog else 0
        cluster1_pou5f1_avg = np.mean(cluster1_pou5f1) if cluster1_pou5f1 else 0
        cluster1_sox2_avg = np.mean(cluster1_sox2) if cluster1_sox2 else 0
        cluster1_ctcf_avg = np.mean(cluster1_ctcf) if cluster1_ctcf else 0

        # Print 
        print ("\nCluster 1 Hist1 percentages:", cluster1_hist1_avg * 100)
        print ("Cluster 1 LAD percentages:", cluster1_lad_avg * 100)
        print ("Cluster 1 VMN percentages:", cluster1_vmn_avg * 100)
        print ("Cluster 1 RNAPII-S2P percentages:", cluster1_rnapii_s2p_avg * 100)
        print ("Cluster 1 RNAPII-S5P percentages:", cluster1_rnapii_s5p_avg * 100)
        print ("Cluster 1 RNAPII-S7P percentages:", cluster1_rnapii_s7p_avg * 100)
        print ("Cluster 1 Enhancer percentages:", cluster1_enhancer_avg * 100)
        print ("Cluster 1 H3K9me3 percentages:", cluster1_h3k9me3_avg * 100)
        print ("Cluster 1 H3K20me3 percentages:", cluster1_h3k20me3_avg * 100)
        print ("Cluster 1 H3K27me3 percentages:", cluster1_h3k27me3_avg * 100)
        print ("Cluster 1 H3K36me3 percentages:", cluster1_h3k36me3_avg * 100)
        print ("Cluster 1 NANOG percentages:", cluster1_nanog_avg * 100)
        print ("Cluster 1 POU5F-7BWU percentages:", cluster1_pou5f1_avg * 100)
        print ("Cluster 1 SOX2 percentages:", cluster1_sox2_avg * 100)
        print ("Cluster 1 CTCF-7BWU percentages:", cluster1_ctcf_avg * 100)

        # Call function for cluster 2 
        (cluster2_hist1, cluster2_lad, cluster2_vmn, cluster2_rnapii_s2p, cluster2_rnapii_s5p, cluster2_rnapii_s7p, cluster2_enhancer, 
        cluster2_h3k9me3, cluster2_h3k20me3, cluster2_h3k27me3, cluster2_h3k36me3, cluster2_nanog, cluster2_pou5f1, cluster2_sox2, cluster2_ctcf) = compute_feature_percentages(
            best_result[2],
            window_np_presence,
            hist1_windows_indices,
            hist1_features,
            lad_features,
            vmn_features,
            rnapii_s2p,
            rnapii_s5p,
            rnapii_s7p,
            enhancer,
            h3k9me3,
            h3k20me3,
            h3k27me3,
            h3k36me3,
            nanog,
            pou5f1,
            sox2,
            ctcf
        )

        #Cluster 2 averages
        cluster2_hist1_avg = np.mean(cluster2_hist1) if cluster2_hist1 else 0
        cluster2_lad_avg = np.mean(cluster2_lad) if cluster2_lad else 0
        cluster2_vmn_avg = np.mean(cluster2_vmn) if cluster2_vmn else 0
        cluster2_rnapii_s2p_avg = np.mean(cluster2_rnapii_s2p) if cluster2_rnapii_s2p else 0
        cluster2_rnapii_s5p_avg = np.mean(cluster2_rnapii_s5p) if cluster2_rnapii_s5p else 0
        cluster2_rnapii_s7p_avg = np.mean(cluster2_rnapii_s7p) if cluster2_rnapii_s7p else 0
        cluster2_enhancer_avg = np.mean(cluster2_enhancer) if cluster2_enhancer else 0  
        cluster2_h3k9me3_avg = np.mean(cluster2_h3k9me3) if cluster2_h3k9me3 else 0
        cluster2_h3k20me3_avg = np.mean(cluster2_h3k20me3) if cluster2_h3k20me3 else 0  
        cluster2_h3k27me3_avg = np.mean(cluster2_h3k27me3) if cluster2_h3k27me3 else 0
        cluster2_h3k36me3_avg = np.mean(cluster2_h3k36me3) if cluster2_h3k36me3 else 0
        cluster2_nanog_avg = np.mean(cluster2_nanog) if cluster2_nanog else 0
        cluster2_pou5f1_avg = np.mean(cluster2_pou5f1) if cluster2_pou5f1 else 0
        cluster2_sox2_avg = np.mean(cluster2_sox2) if cluster2_sox2 else 0
        cluster2_ctcf_avg = np.mean(cluster2_ctcf) if cluster2_ctcf else 0

        # Print
        print ("\nCluster 2 Hist1 percentages:",cluster2_hist1_avg * 100)
        print ("Cluster 2 LAD percentages:",cluster2_lad_avg * 100)
        print ("Cluster 2 VMN percentages:",cluster2_vmn_avg * 100)
        print ("Cluster 2 RNAPII-S2P percentages:",cluster2_rnapii_s2p_avg * 100)
        print ("Cluster 2 RNAPII-S5P percentages:",cluster2_rnapii_s5p_avg * 100)
        print ("Cluster 2 RNAPII-S7P percentages:",cluster2_rnapii_s7p_avg * 100)
        print ("Cluster 2 Enhancer percentages:",cluster2_enhancer_avg * 100)
        print ("Cluster 2 H3K9me3 percentages:",cluster2_h3k9me3_avg * 100)
        print ("Cluster 2 H3K20me3 percentages:",cluster2_h3k20me3_avg * 100)
        print ("Cluster 2 H3K27me3 percentages:",cluster2_h3k27me3_avg * 100)
        print ("Cluster 2 H3K36me3 percentages:",cluster2_h3k36me3_avg * 100)
        print ("Cluster 2 NANOG percentages:",cluster2_nanog_avg * 100)
        print ("Cluster 2 POU5F-7BWU percentages:",cluster2_pou5f1_avg * 100)
        print ("Cluster 2 SOX2 percentages:",cluster2_sox2_avg * 100)
        print ("Cluster 2 CTCF-7BWU percentages:",cluster2_ctcf_avg * 100)

        # Call function for cluster 3
        (cluster3_hist1, cluster3_lad, cluster3_vmn, cluster3_rnapii_s2p, cluster3_rnapii_s5p, cluster3_rnapii_s7p, cluster3_enhancer, 
        cluster3_h3k9me3, cluster3_h3k20me3, cluster3_h3k27me3, cluster3_h3k36me3, cluster3_nanog, cluster3_pou5f1, cluster3_sox2, cluster3_ctcf) = compute_feature_percentages(
            best_result[3],
            window_np_presence,
            hist1_windows_indices,
            hist1_features,
            lad_features,
            vmn_features,
            rnapii_s2p,
            rnapii_s5p,
            rnapii_s7p,
            enhancer,
            h3k9me3,
            h3k20me3,
            h3k27me3,
            h3k36me3,
            nanog,
            pou5f1,
            sox2,
            ctcf
        )

        # Calculate cluster 3 averages
        cluster3_hist1_avg = np.mean(cluster3_hist1) if cluster3_hist1 else 0
        cluster3_lad_avg = np.mean(cluster3_lad) if cluster3_lad else 0
        cluster3_vmn_avg = np.mean(cluster3_vmn) if cluster3_vmn else 0
        cluster3_rnapii_s2p_avg = np.mean(cluster3_rnapii_s2p) if cluster3_rnapii_s2p else 0
        cluster3_rnapii_s5p_avg = np.mean(cluster3_rnapii_s5p) if cluster3_rnapii_s5p else 0
        cluster3_rnapii_s7p_avg = np.mean(cluster3_rnapii_s7p) if cluster3_rnapii_s7p else 0
        cluster3_enhancer_avg = np.mean(cluster3_enhancer) if cluster3_enhancer else 0  
        cluster3_h3k9me3_avg = np.mean(cluster3_h3k9me3) if cluster3_h3k9me3 else 0
        cluster3_h3k20me3_avg = np.mean(cluster3_h3k20me3) if cluster3_h3k20me3 else 0  
        cluster3_h3k27me3_avg = np.mean(cluster3_h3k27me3) if cluster3_h3k27me3 else 0
        cluster3_h3k36me3_avg = np.mean(cluster3_h3k36me3) if cluster3_h3k36me3 else 0
        cluster3_nanog_avg = np.mean(cluster3_nanog) if cluster3_nanog else 0
        cluster3_pou5f1_avg = np.mean(cluster3_pou5f1) if cluster3_pou5f1 else 0
        cluster3_sox2_avg = np.mean(cluster3_sox2) if cluster3_sox2 else 0
        cluster3_ctcf_avg = np.mean(cluster3_ctcf) if cluster3_ctcf else 0

        # Print
        print ("\nCluster 3 Hist1 percentages:", cluster3_hist1_avg * 100)
        print ("Cluster 3 LAD percentages:", cluster3_lad_avg * 100)
        print ("Cluster 3 VMN percentages:", cluster3_vmn_avg * 100)
        print ("Cluster 3 RNAPII-S2P percentages:", cluster3_rnapii_s2p_avg * 100)
        print ("Cluster 3 RNAPII-S5P percentages:", cluster3_rnapii_s5p_avg * 100)
        print ("Cluster 3 RNAPII-S7P percentages:", cluster3_rnapii_s7p_avg * 100)
        print ("Cluster 3 Enhancer percentages:", cluster3_enhancer_avg * 100)
        print ("Cluster 3 H3K9me3 percentages:", cluster3_h3k9me3_avg * 100)
        print ("Cluster 3 H3K20me3 percentages:", cluster3_h3k20me3_avg * 100)
        print ("Cluster 3 H3K27me3 percentages:", cluster3_h3k27me3_avg * 100)  
        print ("Cluster 3 H3K36me3 percentages:", cluster3_h3k36me3_avg * 100)
        print ("Cluster 3 NANOG percentages:", cluster3_nanog_avg * 100)    
        print ("Cluster 3 POU5F-7BWU percentages:", cluster3_pou5f1_avg * 100)
        print ("Cluster 3 SOX2 percentages:", cluster3_sox2_avg * 100)
        print ("Cluster 3 CTCF-7BWU percentages:", cluster3_ctcf_avg * 100)


        # Make a Boxplot for Hist 1
        hist1_data = (
            [(val, "Cluster 1") for val in cluster1_hist1] +
            [(val, "Cluster 2") for val in cluster2_hist1] +
            [(val, "Cluster 3") for val in cluster3_hist1]
        )
        hist1_df = pd.DataFrame(hist1_data, columns=["Percentage", "Cluster"])
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Cluster", y="Percentage", data=hist1_df)
        sns.stripplot(x = "Cluster", y = "Percentage", data = hist1_df, color = "black", alpha = 0.5)
        plt.title("Distribution of Hist1 Feature Percentages by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Percentage of Hist1 Windows Detected")
        plt.tight_layout()
        plt.savefig("hist1_feature_percentages_boxplot.png")
        plt.close()

        # Make a Boxplot for LAD
        lad_data = (
            [(val, "Cluster 1") for val in cluster1_lad] +
            [(val, "Cluster 2") for val in cluster2_lad] +
            [(val, "Cluster 3") for val in cluster3_lad]
        )
        lad_df = pd.DataFrame(lad_data, columns=["Percentage", "Cluster"])
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Cluster", y="Percentage", data=lad_df)
        sns.stripplot(x = "Cluster", y = "Percentage", data = lad_df, color = "black", alpha = 0.5)
        plt.title("Distribution of LAD Feature Percentages by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Percentage of LAD Windows Detected")
        plt.tight_layout()
        plt.savefig("lad_feature_percentages_boxplot.png")
        plt.close()

        # FEATURE SELECTION - 3
        # Calculate percentage of NPs in each radial position for each cluster

        print("\n=== RADIAL POSITION DISTRIBUTION BY CLUSTER ===")
        # Enumerate through each cluster best cluster, starting at 1
        for i, cluster in enumerate([best_result[1], best_result[2], best_result[3]], start=1):
            # Get the radial positions from data computed earlier
            # Radial_position is a list where the index corresponds to the NP index and the value is the radial position (1-5)
            cluster_radial_positions = [radial_position[np_idx] for np_idx in cluster]
            # Print cluster
            print(f"\nCluster {i}:")
            # r is the radial position in this cluster
            count = [cluster_radial_positions.count(r) for r in range(1, 6)]   
            # Total length of cluster for percentage calculation, if cluster is empty set to 1 to avoid division by zero
            total = len(cluster_radial_positions)
            # Calculate percentages and multiply by 100 for actual percentage and avoid division by zero, if total is 0 then set all percentages to 0
            percentages = [(c/total)* 100 for c in count ] if total > 0 else [0] * 5
            print(count)
            print(percentages)
            plt.figure(figsize=(8, 6))
            plt.bar(range(1,6), percentages)
            plt.xticks(range(1,6))
            plt.xlabel("Radial Position")
            plt.ylabel("Percentage of NPs in Cluster")
            plt.title(f"Radial Position Distribution for Cluster {i}")
            plt.tight_layout()
            plt.savefig(f"radial_position_distribution_cluster_{i}.png")
            plt.close()
        
        # FEATURE - SELECTION 4
        #Do the features allow you to discriminate between the clusters?
        #i. Compare the k radar charts.
        #ii. What might be the possible meaning of your findings in the domain of biology?
        # Radar chart for each cluster and feature, with percentage of NPs in that cluster that have that feature

        # Categories is every feature we want to search for
        categories = ["Hist1", "LAD", "VMN", "RNAPII_S2P", "RNAPII_S5P", "RNAPII_S7P", "Enhancer", "H3K9me3", "H3K20me3", "H3K27me3", "H3K36me3", "NANOG", "pou5f1", "sox2", "CTCF"]

        # Multiply by 100 to get percentage for radar chart and create a list of values for cluster 1
        values = [
            cluster1_hist1_avg * 100,
            cluster1_lad_avg * 100,
            cluster1_vmn_avg * 100,
            cluster1_rnapii_s2p_avg * 100,
            cluster1_rnapii_s5p_avg * 100,
            cluster1_rnapii_s7p_avg * 100,
            cluster1_enhancer_avg * 100,
            cluster1_h3k9me3_avg * 100,
            cluster1_h3k20me3_avg * 100,
            cluster1_h3k27me3_avg * 100,
            cluster1_h3k36me3_avg * 100,
            cluster1_nanog_avg * 100,
            cluster1_pou5f1_avg * 100,
            cluster1_sox2_avg * 100,
            cluster1_ctcf_avg * 100,
        ]

        # Number of categories for radar chart         
        n = len(categories)
        values += values[:1]  # Duplicate the values to close the radar chart
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # Duplicate the angles to close the radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Set the direction of the radar chart and the starting point
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Set the category labels on the radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set the y ticks
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
        plt.ylim(0, 100)

        # Plot the radar chart for cluster 1
        ax.plot(angles, values, color = sns.color_palette("deep")[0], linewidth=2, linestyle='solid')
        ax.fill(angles, values, color = sns.color_palette("deep")[0], alpha=0.25)

        # Title and save 
        plt.title("Feature Percentages for Cluster 1", size=20, color=sns.color_palette("deep")[0], y=1.1)
        plt.tight_layout()
        plt.savefig("radar_chart_cluster_1.png")
        plt.close()

        # Repeat for cluster 2
        values = [
            cluster2_hist1_avg * 100,
            cluster2_lad_avg * 100,
            cluster2_vmn_avg * 100,
            cluster2_rnapii_s2p_avg * 100,
            cluster2_rnapii_s5p_avg * 100,
            cluster2_rnapii_s7p_avg * 100,
            cluster2_enhancer_avg * 100,
            cluster2_h3k9me3_avg * 100,
            cluster2_h3k20me3_avg * 100,
            cluster2_h3k27me3_avg * 100,
            cluster2_h3k36me3_avg * 100,
            cluster2_nanog_avg * 100,
            cluster2_pou5f1_avg * 100,
            cluster2_sox2_avg * 100,
            cluster2_ctcf_avg * 100,
        ]
        values += values[:1]  # Duplicate the values to close the radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
        plt.ylim(0, 100)
        ax.plot(angles, values, color = sns.color_palette("deep")[1], linewidth=2, linestyle='solid')
        ax.fill(angles, values, color = sns.color_palette("deep")[1], alpha=0.25)
        plt.title("Feature Percentages for Cluster 2", size=20, color=sns.color_palette("deep")[1], y=1.1)
        plt.tight_layout()
        plt.savefig("radar_chart_cluster_2.png")
        plt.close()
        
        # Repeat for cluster 3
        values = [
            cluster3_hist1_avg * 100,
            cluster3_lad_avg * 100,
            cluster3_vmn_avg * 100,
            cluster3_rnapii_s2p_avg * 100,
            cluster3_rnapii_s5p_avg * 100,
            cluster3_rnapii_s7p_avg * 100,
            cluster3_enhancer_avg * 100,
            cluster3_h3k9me3_avg * 100,
            cluster3_h3k20me3_avg * 100,
            cluster3_h3k27me3_avg * 100,
            cluster3_h3k36me3_avg * 100,
            cluster3_nanog_avg * 100,
            cluster3_pou5f1_avg * 100,
            cluster3_sox2_avg * 100,
            cluster3_ctcf_avg * 100,
        ]
        values += values[:1]  # Duplicate the values to close the radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
        plt.ylim(0, 100)
        ax.plot(angles, values, color = sns.color_palette("deep")[2], linewidth=2, linestyle='solid')
        ax.fill(angles, values, color = sns.color_palette("deep")[2], alpha=0.25)
        plt.title("Feature Percentages for Cluster 3", size=20, color=sns.color_palette("deep")[2], y=1.1)
        plt.tight_layout()
        plt.savefig("radar_chart_cluster_3.png")
        plt.close()

        # In general, if One cluster is high in one of the features, then the other clusters are not 
        # high in that feature, so the features do allow us to discriminate between the clusters.

        # CO-SEGREGATION ACTIVITY
        # Design software to calculate the normalized linkage table containing the normalized linkage for
        # each pair of windows in the Hist1 region.

        # Initialize array
        num_windows = len(hist1_windows_indices)
        normalized_linkage_matrix = np.zeros((num_windows, num_windows))
        # Calculate normalized linkage for each pair of windows
        
        # Loop through windows 
        for i in range(num_windows):
            normalized_linkage_matrix[i][i] = 1  # Set diagonal to 1 since a window is perfectly linked to itself 
            for j in range(i + 1, num_windows):
                w1 = hist1_windows_indices[i]
                w2 = hist1_windows_indices[j]
                # Calculate the number of NPs that detect each window and both windows
                n_w1 = sum(window_np_presence[w1])
                n_w2 = sum(window_np_presence[w2])
                n_both = sum(window_np_presence[w1][np_idx] and window_np_presence[w2][np_idx] for np_idx in range(num_nps))
                
                w1percent = n_w1 / num_nps if n_w1 > 0 else 0
                w2percent = n_w2 / num_nps if n_w2 > 0 else 0
                bothpercent = n_both / num_nps if n_both > 0 else 0

                # Expected change
                expected_change = w1percent * w2percent

                # actual change = bothpercent - expected change
                D = bothpercent - expected_change

                # When Dmax is 0, normalization is undefined so must set linkage to 0 
                if D >= 0:
                    # If D is positive, the maximum positive change is limited by the smaller of the two windows' overlapping percentages
                    # Dmax represents maximum overlap
                    # Want to find the max but also accept we need the minimum to have the overlap 
                    Dmax = min(w1percent * (1 - w2percent), w2percent * (1 - w1percent))
                else:
                    # If D is negative, the maximum negative change is limited by the smaller of the two windows' non-overlapping percentages
                    # Dmax represents maximum non-overlap (serperation) 
                    Dmax = min(w1percent * w2percent, (1 - w1percent) * (1 - w2percent))
                
                # Normalized linkage
                normalized_linkage = D / Dmax if Dmax > 0 else 0
                normalized_linkage_matrix[i][j] = normalized_linkage
                normalized_linkage_matrix[j][i] = normalized_linkage

        # Heatmap for normalized linkage matrix
        generate_heatmap(
            normalized_linkage_matrix,
            "Normalized Linkage Heatmap for Hist1 Windows",
            "Window Index",
            "Window Index",
            "normalized_linkage_heatmap_hist1.png"
        )

        # in this activity, the normalized linkage table is interpreted as a representation of a network by
        # applying the following rules:
        # • The network contains a vertex for each window in the Hist1 region.
        # • The network contains an undirected edge <A,B> when L(A,B) > Q3, where:
        # o L(A,B) = the normalized linkage of windows A and B
        # o Q3 = the Q3 value in the normalized linkage table for the Hist1 region. “The third
        # quartile (Q3), also known as the upper quartile, is the value that separates the
        # top 25% of a dataset from the bottom 75%. It's the median of the upper half of
        # the data, representing the 75th percentile.” (Source: Google AI summary)
        # o A is not equal to B (i.e., there are no reflexive edges in the graph)

        # Create graph and add edges based on normalized linkage values
        G = nx.Graph()

        # Get the upper triangle values (excluding the diagonal) for Q3 calculation
        # Such as the top right of a symmetric matrix, we only want to consider each pair once and not the diagonal
        upper_triangle_values = normalized_linkage_matrix[np.triu_indices(num_windows, k=1)]
        
        Q3 = np.percentile(upper_triangle_values, 75)
        G.add_nodes_from(range(num_windows))  # Add nodes for each window
        for i in range(num_windows):
            for j in range(i + 1, num_windows):
                # See if 2 edges interact stongly  
                if normalized_linkage_matrix[i][j] >= Q3:
                    G.add_edge(i, j)

        centrality = nx.degree_centrality(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True)
        plt.title("Network Graph of Hist1 Windows Based on Normalized Linkage")
        plt.savefig("hist1_windows_network_graph.png")
        plt.close()

        # Print stats 
        print("\n=== Normalized Linkage Network Graph Stats ===")
        values = list(centrality.values())
        print("Average degree centrality:", np.mean(values))
        print("Max degree centrality:", np.max(values))
        print("Min degree centrality:", np.min(values))
        # Print degree centrality for each node sorted by value (Ascending)
        for node, val in sorted(centrality.items(), key = lambda x: x[1]):
            print(f"Window {node}: Degree Centrality = {val}")

        # Find the 5 nodes with the highest degree centrality
        top_5 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print ("\nTop 5 windows with highest degree centrality:")
        for node, val in top_5:
            print(f"Window {node}: Degree Centrality = {val}")
        
        # Find the neighbors of the top 5 nodes with the highest degree centrality
        # Print size of each neighbors list
        # Print percentage of nodes in community that contain hist1 genes
        # Print percentage of nodes in community that contain LAD 


        # The top 5 windows with the highest degree centrality are the biggest hubs of chromatin interactions in the hist 1 region 
        # By looking at the neighbors of these windows, we can see which other windows they interact with and if there are any patterns in the features of those neighbors.

    
        print("\n=== Neighbors of Top 5 Windows with Highest Degree Centrality ===")
        # Get lad window indices and features for percentage calculation
        lad_features = df["LAD"].astype(int).tolist()
        lad_window_indices = [i for i, val in enumerate(lad_features) if val == 1]

        # loop through nodes in the top 5 windows with the highest degree centrality and print their neighbors and features
        for node, val in top_5:
            # Compute neighbors plus node
            neighbors = list(G.neighbors(node)) + [node]  

            print(f"\nWindow {node}:")
            print(f"Degree Centrality: {val}")
            print(f"Number of Neighbors: {len(neighbors)}")
            print(f"Neighbors: {neighbors}")    

            # Calculate percentage of neighbors that contain hist1 genes
            hist1_neighbors = sum(1 for neighbor in neighbors if hist1_features[neighbor] == 1)
            lad_neighbors = sum(1 for neighbor in neighbors if lad_features[neighbor] == 1)
            total_neighbors = len(neighbors)
            hist1_percentage = (hist1_neighbors / total_neighbors) * 100 if total_neighbors > 0 else 0
            lad_percentage = (lad_neighbors / total_neighbors) * 100 if total_neighbors > 0 else 0
            print(f"Percentage of Neighbors with Hist1 genes: {hist1_percentage:.2f}%")
            print(f"Percentage of Neighbors with LAD: {lad_percentage:.2f}%")

            # LAD enrichment can indicate that the community of windows around this node is more likely to be located in the nuclear periphery, which is often associated with gene repression.
            # These are regions attatched to the outer edge of the nucleus and are generally less active or repressed regions of the genome. 
            # However, they are strong structural components of the genome and can play important roles in organizing the 3D structure of the genome and regulating gene expression.

            # Visualize the community as a graph (node is genomic window), size of node is proportionial to 
            # degree centrality (use already computed degree centrality
            # Edge represents an interaction between windows

            # crete a subgraph of the neighbors and the node itself to visualize the community around that node
            subgraph = G.subgraph(neighbors)
            plt.figure(figsize=(8, 6))
            # pos is the position of the nodes in the graph
            pos = nx.spring_layout(subgraph)
            # compute node sizes by multiplying degree centrality by a constant to make the nodes visible
            # we can use the already computed degree centrality for the whole graph and just get the values for the neighbors and node itself
            node_sizes = [centrality[neighbor] * 1000 for neighbor in subgraph.nodes()]
            nx.draw(subgraph, pos, with_labels=True, node_size=node_sizes)
            plt.title(f"Community Graph for Window {node} and its Neighbors")
            plt.savefig(f"community_graph_window_{node}.png")
            plt.close()

            # Visualize the community as a heatmap 
            # 81 x 81
            # Each cell is one edge in the graph
            # Heatmap should show only the subgraph that corresponds to the community 

            # Create a submatrix of the normalized linkage matrix that corresponds to the neighbors and the node itself
            subgraph_matrix = np.zeros((num_windows, num_windows))
            for i in neighbors:
                for j in neighbors:
                    subgraph_matrix[i][j] = normalized_linkage_matrix[i][j]

            generate_heatmap(
                subgraph_matrix,
                f"Community Heatmap for Window {node} and its Neighbors",
                "Neighbor Index",
                "Neighbor Index",
                f"community_heatmap_window_{node}.png"
            )

            # All heatmaps look fairly similar because of a lot of neighbors are shared between multiple communities meaning that the communities are not very distinct from each other. 
            # However, we can see that some communities have more strong interactions (higher normalized linkage values) than others, 
            # which could indicate that those communities are more tightly connected and may represent more functionally related regions of the genome.


            # In our communities, one node is in multiple communities which biologically it means that one genomic window can be part of multiple different interaction communities, 
            # which is consistent with the complex and dynamic nature of chromatin interactions in the nucleus.
            # A high LAD window in multiple communites could be the anchor point for multiple different chromatin interactions at the nuclear periphery,
            # which could be important for organizing the 3D structure of the genome and regulating gene expression in those regions.
            

    
        

        
    
if __name__ == "__main__":
    main()
    # testing
    # test from mac

