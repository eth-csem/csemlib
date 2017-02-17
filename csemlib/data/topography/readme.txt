Original data is contained in 10MinuteTopoGrid.txt, this file has been resampled to a 20 minute grid

#new_sampling is defined as:
start = 1
col = np.linspace(-90 + start, 90 - start, 179)
lon = np.linspace(-180 + start, 180 - start, 359)

script used for resampling:
        # initial_reading = False
        # if initial_reading:
        #     # Initial Reading
        #     vals = np.genfromtxt(os.path.join(self.directory, '10MinuteTopoGrid.txt'), delimiter=',')
        #     _, _, topo = vals.T
        #
        #     initial_lon = np.linspace(-180, 180, 360 * 6 + 1)
        #     initial_col = np.linspace(-90, 90, 180 * 6 + 1)
        #
        #     topo = np.array(topo)
        #     topo_reshaped = topo.reshape(len(initial_col), len(initial_lon))
        #
        #     # Resample such that there are no points at the poles
        #     topo_resampled = topo_reshaped[6:-6:6, 6:-6:6]
        #     topo_1d = topo_resampled.reshape(np.size(topo_resampled))
        #     np.savetxt(os.path.join(self.directory, 'topo_resampled.txt'), topo_1d, fmt='%.0f')

