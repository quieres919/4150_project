import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def is_present(value):
    return value != "0"

def main():
    filename = "GSE64881_segmentation_at_30000bp.passqc.multibam (2).txt"

    with open(filename, "r") as f:
        # Store coords in list for later use
        window_coords = []
        #header
        window_np_presence = []
        header = f.readline().strip()
        columns = header.split()

        # Columns 0,1,2 = chrom, start, stop
        np_headers = columns[3:]

        # Count NP columns (those whose header contains "F")
        num_nps = 0
        for h in np_headers:
            if "F" in h:
                num_nps += 1

        num_windows = 0
        windows_per_np = [0] * num_nps
        total_nps_per_window = 0
        min_nps_per_window = None
        max_nps_per_window = 0

        # Window frequency data for compaction estimate
        window_detection = []

        #Now read the *data* lines
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3 + num_nps:
                # skip malformed line
                continue

            chrom = parts[0]
            start = int(parts[1])
            stop = int(parts[2])
            window_coords.append((chrom, start, stop))

            num_windows += 1

            # everything after chrom/start/stop
            np_values = parts[3:]
            nps_this_window = 0
            presence = []

            for idx in range(num_nps):
                val = np_values[idx]
                detected = is_present(val)
                presence.append(detected)

                if detected:
                    windows_per_np[idx] += 1
                    nps_this_window += 1

            window_np_presence.append(presence)

            total_nps_per_window += nps_this_window

            if min_nps_per_window is None or nps_this_window < min_nps_per_window:
                min_nps_per_window = nps_this_window
            if nps_this_window > max_nps_per_window:
                max_nps_per_window = nps_this_window

            # Store detection frequency for this window
            window_detection.append(nps_this_window / num_nps)

        #Calculate detection frequency per NP
        detection_frequencies = [w / num_windows for w in windows_per_np]
        
        #Calculate mean and solve for good threshhold for outliers
        mean_freq = np.mean(detection_frequencies)
        std_freq = np.std(detection_frequencies)
        outlier_threshold = mean_freq + 2 * std_freq
        
        #Calculate radial positions based on percentiles
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
            
        # Output results
        for idx, freq in enumerate(detection_frequencies):
            outlier_tag = "OUTLIER HIGH" if freq > outlier_threshold else ""
            #print only outliers
            if outlier_tag:
                print(f"NP {idx+1}: Detection Frequency = {freq:.4f} {outlier_tag}")
            
        window_array = np.array(window_detection)
        # Start finding the estimated compaction based on frequency (Basically inverse of radial position with more bins)
        # Lower frequency = higher compaction
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
            # Check if its between start and stop 
            if chrom == hist1_chrom and not (stop < hist1_start or start > hist1_stop):
                # Count windows overlapping Hist1 region
                hist1_windows_indices.append(w_idx)
                # Count Nps detecting Hist1 region / reset counter
                nps_detecting = 0
                for np_idx in range(num_nps):
                    # Check if NP detected this window
                    if window_np_presence[w_idx][np_idx]:
                        nps_detecting += 1
                        hist1_np_indices.add(np_idx)
                        windows_per_np_hist1[np_idx] += 1
                nps_per_windows_hist1.append(nps_detecting)

        print ("\n=== HIST1 REGION ANALYSIS ===")
        print("Numer of genomic windows: ", len(hist1_windows_indices))
        print("Number of Nps detecting Hist1:", len(hist1_np_indices))
        # Windows per NP
        print("Windows per NP:")
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
        # Check values 
        #print("\n=== JACCARD INDEX FOR RELEVANT NPs ===")
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
        
        # Convert hist1_np_indices to a sorted list
        hist1_np_indices_sorted = sorted(hist1_np_indices)  # or sort by windows detected for trend

        # --- Similarity Heatmap ---
        sim_subset = jaccard_index_matrix[np.ix_(hist1_np_indices_sorted, hist1_np_indices_sorted)].copy()

        # Make symmetric and set diagonal = 1
        i_lower = np.tril_indices_from(sim_subset, -1)
        sim_subset[i_lower] = sim_subset.T[i_lower]
        np.fill_diagonal(sim_subset, 1)

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(sim_subset, cmap="coolwarm")
        plt.title("Jaccard Similarity Heatmap for HIST1 NPs")
        step = 20
        ticks = list(range(0, len(hist1_np_indices), step))
        plt.xlabel("NP Index")
        plt.ylabel("NP Index")
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.savefig("jaccard_similarity_heatmap_hist1.png")
        plt.close()
        #plt.show()

        # --- Distance Heatmap ---
        dist_subset = jaccard_distance_matrix[np.ix_(hist1_np_indices_sorted, hist1_np_indices_sorted)].copy()

        # Make symmetric and set diagonal = 0
        i_lower = np.tril_indices_from(dist_subset, -1)
        dist_subset[i_lower] = dist_subset.T[i_lower]
        np.fill_diagonal(dist_subset, 0)

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(dist_subset, cmap="viridis")
        plt.title("Jaccard Distance Heatmap for HIST1 NPs")
        plt.xlabel("NP Index")
        plt.ylabel("NP Index")
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.savefig("jaccard_distance_heatmap_hist1.png")
        plt.close()
        #plt.show()
        ## Assignment 6: Clustering
        # Normalize our matrices
        random_seeds = np.random.choice(hist1_np_indices, size=3, replace=False)
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
                    if window_np_presence[w_idx][np1]:
                        windows_np1.add(w_idx)
                    if window_np_presence[w_idx][np2]:
                        windows_np2.add(w_idx)
                intersection = len(windows_np1.intersection(windows_np2))   
                A_union = len(windows_np1)
           
                B_union = len(windows_np2)
            
            # Jaccard similarity normalized
            jacard_similarity_normalized = intersection / min(A_union, B_union) if min(A_union, B_union) > 0 else 0
            jaccard_similarity_normalized_matrix[np1][np2] = jacard_similarity_normalized
            jaccard_distance_normalized_matrix[np2][np1] = jacard_similarity_normalized
            # Jaccard distance normalized 
            jaccard_distance_normalized = 1 - jacard_similarity_normalized
            jaccard_distance_normalized_matrix[np1][np2] = jaccard_distance_normalized
            jaccard_distance_normalized_matrix[np2][np1] = jaccard_distance_normalized
        # Heatmaps for normalized matrices can be created similarly if needed
        sim_norm_subset = jaccard_similarity_normalized_matrix[np.ix_(hist1_np_indices_sorted, hist1_np_indices_sorted)].copy()
        i_lower = np.tril_indices_from(sim_norm_subset, -1)
        import random
        k = 3
        all_indices = list(hist1_np_indices_sorted)
        initial_centers = random.sample(all_indices, k)
        cluster_assignments = np.zeros(len(all_indices), dtype=int)

        for i, np_idx in enumerate(all_indices):
            distances = [
                jaccard_distance_normalized_matrix[np_idx][center] for center in initial_centers
            ]
            cluster_assignments[i] = np.argmin(distances)
        print("Initial cluster centers:", initial_centers)
        print("Cluster assignments of NPs:", cluster_assignments)
        print("Num clusters assigned:", np.bincount(cluster_assignments))


        sim_norm_subset[i_lower] = sim_norm_subset.T[i_lower]
        np.fill_diagonal(sim_norm_subset, 1)
        plt.figure(figsize=(10, 10))
        ax =  sns.heatmap(sim_norm_subset, cmap="coolwarm")
        step = 20
        ticks = list(range(0, len(hist1_np_indices), step))
        plt.title("Normalized Jaccard Similarity Heatmap for HIST1 NPs")
        plt.xlabel("NP Index")
        plt.ylabel("NP Index")
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.savefig("jaccard_similarity_normalized_heatmap_hist1.png")
        plt.close()
        # Heatmap for normalized distance matrix
        dist_norm_subset = jaccard_distance_normalized_matrix[np.ix_(hist1_np_indices_sorted, hist1_np_indices_sorted)].copy()
        i_lower = np.tril_indices_from(dist_norm_subset, -1)
        dist_norm_subset[i_lower] = dist_norm_subset.T[i_lower]
        np.fill_diagonal(dist_norm_subset, 0)
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(dist_norm_subset, cmap="viridis")
        plt.title("Normalized Jaccard Distance Heatmap for HIST1 NPs")
        plt.xlabel("NP Index")
        plt.ylabel("NP Index")
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.savefig("jaccard_distance_normalized_heatmap_hist1.png")
        plt.close()

if __name__ == "__main__":
    main()
    # testing
    # test from mac