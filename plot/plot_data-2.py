__author__ = 'jhaux'

import plot_fingering as plf
import jimlib as jim
import numpy as np

# global handles:


path_to_files    = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/630_nm'
path_to_fingers  = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/intensities_630.csv'
path_to_plot     = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/plot_all.pdf'
path_to_fft      = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/fft.pdf'
path_to_k        = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/k_space.pdf'
path_to_drifter  = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/drifter.pdf'
globmm_savename  = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/glob_mm.csv'
dark_savename    = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/dark.csv'
diff_intensities = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/diff_intensities.csv'
quot_intensities = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/quot_intensities.csv'
diff_lengths     = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/diff_lengths.csv'
quot_lengths     = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/quot_lengths.csv'

# Store all filenames in one array
all_Files = jim.listdir_nohidden(path_to_files)

if True:
    # works only if plf.save_fingerdata() has been run once:
    finger_data = np.genfromtxt(quot_lengths)
    timesteps, fingers, pixels, meters = plf.format_data(finger_data)  # [start:end][::step])


# Data selection

patch = (443, 1500, 1443, 2150)
xmin, xmax, ymin, ymax = patch
cuts=(0, 0, 0, int((ymax - ymin) / 1.35))

normpatch=(1190, 1310,  60, 160)
normcrit = 'linear'

#reference frame
ref_n = 0

# Set rescaling limits
low_d, high_d = 0., 100.
low_q, high_q = 30., 70.


# Set fft parameters:
handle = 2  # cm^1
clean_crit = 'wavelength'
cell_width = 27.3  #cm


if False:
    # safe darkframe and global min/max
    plf.save_darkframe(dark_savename, path_to_files)
    plf.save_glob_minmax(globmm_savename, all_Files=all_Files, dark=np.genfromtxt(dark_savename),normpatch=normpatch)


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
        patch=patch,
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
if True:
    plf.plot_all(
        all_Files,
        path_to_files,
        savename=path_to_plot,
        gauss=True,
        show_finger_bars=False,
        plot_q=False,
        plot_d=True,
        plot_r=False,
        from_file=True,
        dark_savename=dark_savename,
        globmm_savename=globmm_savename,
        gauss_sigma=2,
        gauss_order=0,
        ref_n=ref_n,
        N=10,
        start=1,
        last=200,
        low_d=low_d,
        high_d=high_d,
        low_q=low_q,
        high_q=high_q,
        bar_w=10,
        patch=patch,
        normpatch=normpatch,
        normcrit=normcrit,
        cuts=cuts,
        fingerlengths=fingers,
        timesteps=timesteps
    )


p_start, p_stop, p_step = 1, 50, None
alpha = 0.8
cmap = 'cool'
l=10
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
                figsize=(l*1,l*1)
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
                         stop=None,
                         low_q=low_q,
                         high_q=high_q,
                         from_file=True,
                         gauss=False,
                         figsize=(10,15),
                         add_length=2)