from scipy.interpolate import Rbf
import numpy as np
import matplotlib.pyplot as plt

class Analyzer():
    def __init__(self, labels, patients, masks):
        self.coords = np.array([        #octopus 900 coordinates
    (0,0), (1,-1), (-1,-1), (-1,1), (1,1), (2,-2), (-2,-2), (-2,2),
    (2,2), (4,-1), (4,-4), (1,-4), (-1,-4), (-4,-4), (-4,-1), (-4,1), (-4,4),
    (-1,4), (1,4), (4,4), (4,1), (6,0), (6,-6), (2,-6), (-2,-6), (-6,-6),
    (-6,-1), (-6,1), (-6,6), (-2,6), (2,6), (6,6), (8,-1), (8,-6), (8,-8), (6,-8), (2,-8), 
    (-2,-8), (-6,-8), (-8,-8), (-8,-6), (-8,-1), (-8,1), (-8,6), (-8,8), (-6,8), (-2,8), (2,8), (6,8),
    (8,8), (8,6), (8,1), (10,-3), (3,-10), (-3,-10), (-10, -2), (-10,2), (-3,10), (3,10), (10,3)
    ])
        self.patients = patients
        self.masks = masks
        self.labels = labels
        self.x_label, self.counts = np.unique(self.labels, return_counts=True)    #label of each cluster and num in each
        self.clusters = {}
        for c in np.unique(labels):
            self.clusters[c] = np.where(labels == c)[0]

        self.n_clusters = len(self.clusters)
        self.N, self.T, self.P = self.patients.shape

    def _plot_interpolated_vf(self, values, ax=None, cmap='viridis', title=None, vmin=None, vmax=None):
        """
        values: (60,) vector
        ax: matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))

        grid_x, grid_y = np.mgrid[-12:12:200j, -12:12:200j] # interpolation grid

        rbf = Rbf(self.coords[:,0], self.coords[:,1], values, function='thin_plate')
        grid_z = rbf(grid_x, grid_y)

        radius = 12
        mask = np.sqrt(grid_x**2 + grid_y**2) <= radius     # circular mask
        masked_grid = np.where(mask, grid_z, np.nan)

        image = ax.imshow(masked_grid.T, extent=(-12, 12, -12, 12), origin='lower', cmap=cmap,
                        vmin = vmin, vmax = vmax) #plot

        ax.set_title(title if title else "")
        ax.axis('off')
        
        return image
    
    def _plot_interp_feature(self, feature, cmap = 'viridis', title = ''):
        fig, axes = plt.subplots(1, self.n_clusters, figsize=(4*self.n_clusters, 5))

        all_vals = np.concatenate([feature[c] for c in self.clusters])
        vmin, vmax = all_vals.min(), all_vals.max()

        for ax, c in zip(axes, self.clusters):
            values = feature[c]
            im = self._plot_interpolated_vf(values, ax=ax,
                                    cmap=cmap, title=f"Cluster {c}", vmin=vmin, vmax=vmax)

        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        fig.suptitle(title, fontsize=25)
        return plt

    def show_label_distribution(self):
        plt.figure(figsize=(6,4))
        plt.bar(self.x_label, self.counts, color='grey')
        plt.xlabel("Cluster")
        plt.ylabel("Number of Patients")
        plt.title("Patient Distribution Across Clusters")
        plt.xticks(self.x_label)
        plt.savefig('unsupervised_model/label_distribution.png')

    def show_visit_distribution(self):
        avg_visits = []
        for c in self.x_label:
            i = self.clusters[c]     # compute average visits per cluster
            avg_visits.append(self.masks[i].sum(axis=1).mean())

        plt.figure(figsize=(6,4))
        plt.bar(self.x_label, avg_visits, color='lightblue')
        plt.xlabel("Cluster")
        plt.ylabel("Number of Patients")
        plt.title("Average Number of Visits per Patient by Cluster")
        plt.xticks(self.x_label)
        plt.savefig('unsupervised_model/visit_distribution.png')

    def show_mean_baseline(self):
        self.baseline_means = {c: np.nanmean(self.patients[idx, 0, :], axis=0)
                          for c, idx in self.clusters.items()}
        
        plot = self._plot_interp_feature(self.baseline_means, cmap='viridis', title = 'Mean Baseline VF per Cluster')
        plot.savefig('unsupervised_model/mean_baseline.png')

    def _last_visit_per_patient(self):
        last_visits = np.full((self.N, self.P), np.nan, dtype=float)
        for i in range(self.N):
            valid_idx = np.where(self.masks[i] == 1)[0]
            if valid_idx.size > 0:
                last_idx = valid_idx[-1]
                last_visits[i] = self.patients[i, last_idx]
        return last_visits

    def show_mean_final(self):
        last_visits = self._last_visit_per_patient()

        self.last_means = {c: np.nanmean(last_visits[idx], axis=0)
                           for c, idx in self.clusters.items()}
        
        plot = self._plot_interp_feature(self.last_means, cmap='viridis', title = 'Mean Final VF per Cluster')
        plot.savefig('unsupervised_model/mean_final.png')

    def show_mean_change(self):
        diff_means = {c: self.last_means[c]- self.baseline_means[c] for c in self.clusters.keys()}
    
        plot = self._plot_interp_feature(diff_means, cmap='coolwarm_r', title = 'Mean Change in VF by Cluster')
        plot.savefig('unsupervised_model/mean_change.png')

    def show_mean_prog_slope(self):
        mean_slope = np.zeros((self.N, self.P))

        for i in range(self.N):
            valid_visits = self.patients[i][self.masks[i]==1]  #real visits only
            diffs = np.diff(valid_visits, axis=0)  # difference between consecutive visits
            mean_slope[i] = np.mean(diffs, axis=0)  # average slope per location

        cluster_slope_vf = {}
        for c, idx in self.clusters.items():
            cluster_slope_vf[c] = np.mean(mean_slope[idx], axis=0)

        plot = self._plot_interp_feature(mean_slope, cmap='coolwarm_r', title = 'Mean Progression Slope per Cluster')
        plot.savefig('unsupervised_model/mean_slope.png')

    def run_all(self):
        '''
        Creates all plots
        '''
        print('Label Distribution...')
        self.show_label_distribution()
        print('Visit Distribution...')
        self.show_visit_distribution()
        print('Mean Baseline...')
        self.show_mean_baseline()
        print('Mean Final...')
        self.show_mean_final()
        print('Mean Change...')
        self.show_mean_change()
        print('Mean Progression Slope...')
        self.show_mean_prog_slope()
            
        print('Done')
            




    


        


        
        
        

    

    
    