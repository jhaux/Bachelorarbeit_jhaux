__author__ = 'jhaux'

import plot_fingering as plf
import ltm_waves as ltw
import jimlib as jim
import numpy as np
import cv2

# global handles:


path_to_files    = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/630_nm'
path_to_fingers  = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/intensities_630.csv'
path_to_plot     = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/plot_all.pdf'
path_to_fft      = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/fft.pdf'
path_to_k        = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/k_space.pdf'
path_to_drifter  = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/drifter.pdf'
path_to_fcount   = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/fcount.pdf'
path_to_fgrowth  = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/fgrowth.pdf'
globmm_savename  = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/glob_mm.csv'
dark_savename    = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/dark.csv'
diff_intensities = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/diff_intensities.csv'
quot_intensities = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/quot_intensities.csv'
diff_lengths     = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/diff_lengths.csv'
quot_lengths     = u'/home/johannes/Google Drive/Uni HD/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/quot_lengths.csv'

# Store all filenames in one array
all_Files = jim.listdir_nohidden(path_to_files)

if True:
    # works only if plf.save_fingerdata() has been run once:
    finger_data = np.genfromtxt(quot_lengths)
    timesteps, fingers, pixels, meters = plf.format_data(finger_data)  # [start:end][::step])


# Data selection
patch = (643, 1779, 1545, 2000)
xmin, xmax, ymin, ymax = patch
cuts=(0, 0, 0, int((ymax - ymin) / 2))
extreme_cuts=(0, 0, int(0.05*(ymax - ymin)), int(0.9*(ymax - ymin)))
patch_c = (xmin, xmax, ymin, ymax - int((ymax - ymin) / 2))

center, span = 935, 100
single_patch = (center-span/2, center+span/2, ymin-50, ymax)

moved_patch = (xmin, xmax, ymin-50, ymax+50)
normpatch=(1190, 1310, 60, 160)
normcrit = 'linear'

#reference frame
ref_n = 0

# Set rescaling limits
low_d, high_d = 55., 80.
low_q, high_q = 67.5, 75.


# Set fft parameters:
handle = 2  # cm^1
clean_crit = 'wavelength'
cell_width = 27.3  #cm

# Get total picture limits:
y_totalmax, x_totalmax = jim.rotate_90(np.mean(cv2.imread(all_Files[0]), axis=-1), 0).shape
print x_totalmax, y_totalmax

if False:
    # safe darkframe and global min/max
    plf.save_darkframe(dark_savename, path_to_files)
    plf.save_glob_minmax(all_Files=all_Files, globmm_savename=globmm_savename, dark=np.genfromtxt(dark_savename), normpatch=normpatch)


if False:
    # Save that data babe!
    plf.save_raw_intensities(
        all_Files,
        path_to_files,
        doutfile=diff_intensities,
        qoutfile=quot_intensities,
        save_q=True,
        save_d=True,
        from_file=True,
        dark_savename=dark_savename,
        globmm_savename=globmm_savename,
        ref_n=ref_n,
        low_d=low_d,
        high_d=high_d,
        low_q=low_q,
        high_q=high_q,
        patch=patch,
        normpatch=normpatch,
        normcrit=normcrit,
        cuts=(0,0,0,0)
    )

# get those fingers
if False:
    plf.save_finger_lengths(
        doutfile=diff_lengths,
        qoutfile=quot_lengths,
        all_Files=all_Files,
        dark_savename=dark_savename,
        globmm_savename=globmm_savename,
        patch=patch_c,
        normpatch=normpatch,
        normcrit=normcrit,
        ref_n=ref_n,
        low_d=low_d,
        high_d=high_d,
        low_q=low_q,
        high_q=high_q,
        from_file=True,
        save_d=True,
        save_q=True,
        q_threshold=30,
        d_threshold=40,
        handle=handle,
        clean_criterium=clean_crit,
        cell_width=cell_width
    )

# Plot finger evolution:
if False:
    alle = 10
    n_pages = 1
    startframe = 0
    for i in np.arange(n_pages):
        if (i+1)%2 == 0:
            cbar_loc = 'right'
        else:
            cbar_loc = 'left'
        plf.plot_all(
            all_Files,
            path_to_files,
            savename=path_to_plot[:-4] + '_overview_' + str(i) + '.pdf',
            gauss=True,
            show_finger_bars=False,
            plot_q=False,
            plot_d=False,
            plot_r=True,
            from_file=True,
            absolute_time=False,
            dark_savename=dark_savename,
            globmm_savename=globmm_savename,
            gauss_sigma=2,
            gauss_order=0,
            ref_n=ref_n,
            N=10,
            start=i*alle/n_pages + startframe,
            last=(i + 1) * alle/n_pages,
            low_d=low_d,
            high_d=high_d,
            low_q=low_q,
            high_q=high_q,
            bar_w=10,
            patch=moved_patch,
            normpatch=normpatch,
            normcrit=normcrit,
            cuts=extreme_cuts,
            fingerlengths=fingers,
            timesteps=timesteps,
            cbar_loc=cbar_loc,
            x_totalmax=x_totalmax,
            cell_width=cell_width

        )

# Plot finger evolution:
if False:
    alle = len(all_Files) - 1
    for i in np.arange(6):
        if (i+1)%5 == 0:
            make_cbar=True
            cbar_loc = 'bottom'
        else:
            make_cbar=False
            cbar_loc = 'right'
        plf.plot_patch(
            all_Files,
            path_to_files,
            savename=path_to_plot[:-4] + '_overview_' + str(i) + '.pdf',
            gauss=True,
            show_finger_bars=False,
            plot_q=False,
            plot_d=False,
            plot_r=True,
            plot_adjusted=False,
            from_file=True,
            absolute_time=False,
            name_in_x=False,
            dark_savename=dark_savename,
            globmm_savename=globmm_savename,
            gauss_sigma=2,
            gauss_order=0,
            ref_n=ref_n,
            N=3,
            start=i*alle/6 + 5,
            last=(i + 1) * alle/6,
            low_d=low_d,
            high_d=high_d,
            low_q=low_q,
            high_q=high_q,
            bar_w=10,
            patch=moved_patch,
            normpatch=normpatch,
            normcrit=normcrit,
            cuts=(0,0,0,0),
            fingerlengths=fingers,
            timesteps=timesteps,
            make_cbar=make_cbar,
            cbar_loc=cbar_loc,
            x_totalmax=x_totalmax,
            cell_width=cell_width
        )


p_start, p_stop, p_step = 0, 24, None
alpha = 0.8
cmap = 'cool'
l=14
if False:
    all_intensities = np.genfromtxt(quot_intensities)[:,1:]  # 1: da dt vorne dran
    plf.plot_fft(savename=path_to_fft,
                all_intensities=all_intensities,
                start=p_start,
                stop=p_stop,
                step=p_step,
                handle=handle,
                clean_criterium=clean_crit,
                cell_width=cell_width,
                cmap_name=cmap,
                xlims=(0,cell_width),
                ylims=(0,50),
                alpha=alpha,
                figsize=(l*1,l*0.5)
    )

if False:
    all_intensities = np.genfromtxt(quot_intensities)[:,1:]  # 1: da dt vorne dran
    plf.plot_wavelengthspace(savename=path_to_k,
                all_intensities=all_intensities,
                start=p_start,
                stop=p_stop,
                step=p_step,
                handle=handle,
                clean_criterium=clean_crit,
                cell_width=cell_width,
                cmap_name=cmap,
                xlims=(0,handle),
                ylims=(0,3000),
                alpha=alpha,
                figsize=plf.cm2inch((l*1,l*1))
    )

if False:
    plf.plot_fingerdrift(savename=path_to_drifter,
                         all_Files=all_Files,
                         patch=patch,
                         normpatch=normpatch,
                         globmm_savename=globmm_savename,
                         dark_savename=dark_savename,
                         line=50,
                         ref_n=ref_n,
                         start=0,
                         stop=80,
                         low_q=low_q,
                         high_q=high_q,
                         from_file=True,
                         gauss=False,
                         figsize=(10,15),
                         add_length=5)

if False:
    plf.plot_fingercount(path_to_fcount, fingers=fingers, times=timesteps)

if False:
    plf.plot_fingergrowth(path_to_fgrowth, fingers=fingers, times=timesteps, cell_width=cell_width, x_length=xmax-xmin)

if True:
    length = xmax - xmin
    px2cm = cell_width/float(length)
    # parts=(0, length)
    parts = np.arange(100, xmax-xmin-100, step=100)
    ltw.plot_finger_growth(path_to_fgrowth, finger_data, parts, px2cm=px2cm, all_Files=all_Files, xlims=(1, 39) )