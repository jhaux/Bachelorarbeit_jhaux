__author__ = 'jhaux'

import jimlib as jim
import image_operations as imop
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np
import cv2
import datetime


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def quotient_loc(image, reference):
    """Final Version of the quotient method, as described in the Thesis"""
    image -= image.min()
    image *= 100. / float(image.max())

    ref = np.copy(reference)
    ref -= ref.min()
    # print reference.max()
    ref *= 100. / float(ref.max())

    ref[ref < 0.000001] = np.inf  # divide by zero returns 0 => makes sense, because where the reference is dark also
    # the image must be dark!
    # print ref.min(), ref[ref < 101].max()  # debugging

    quot = np.divide(image, ref)

    quot -= quot.min()
    # print quot.min(), quot.max()  # debugging
    quot *= 100. / quot.max()

    return quot


def quotient(image, reference, glob_min, glob_max):
    """Final Version of the quotient method, as described in the Thesis"""
    image -= glob_min
    image *= 100. / float(glob_max - glob_min)

    ref = np.copy(reference)
    ref -= glob_min
    # print reference.max()
    ref *= 100. / float(glob_max - glob_min)

    ref[ref < 0.000001] = np.inf
    # divide by zero returns 0 => makes sense, because where the reference is dark also
    # the image must be dark!

    # print ref.min(), ref[ref < 101].max()  # debugging

    quot = np.divide(image, ref)

    quot -= quot.min()
    # print quot.min(), quot.max()  # debugging
    quot *= 100. / quot.max()

    return quot


def difference(image, reference, glob_min, glob_max):
    """Final Version of the quotient method, as described in the Thesis"""
    image -= glob_min
    image *= 100. / float(glob_max - glob_min)

    ref = np.copy(reference)
    ref -= glob_min
    # print reference.max()
    ref *= 100. / float(glob_max - glob_min)

    # ref[ref < 0.000001] = np.inf  # divide by zero returns 0 => makes sense, because where the reference is dark also
    # the image must be dark!
    # print ref.min(), ref[ref < 101].max()  # debugging

    diff = image - ref

    diff -= diff.min()
    # print quot.min(), quot.max()  # debugging
    diff *= 100. / diff.max()

    return diff


def rescale(image, low=0, top=100):
    image[image > top] = top
    image[image < low] = low

    image -= image.min()
    image *= 100. / image.max()

    return image


def format_data(all_data, cell_width=0.23):
    '''Standard operations to get the wanted information out of the previously stored wavelength-data files'''
    if len(all_data.shape) > 1:
        timesteps = all_data[:, 0]
        intensities = all_data[:, 1:]
        pixels = np.arange(len(all_data[0, 1:]))
        meters = np.linspace(0, cell_width, num=len(intensities[0]))
    else:
        timesteps = all_data[0]
        intensities = all_data[1:]
        pixels = np.arange(len(all_data[1:]))
        meters = np.linspace(0, cell_width, num=len(intensities))

    return timesteps, intensities, pixels, meters


def dt2hms(t, t_0):
    t = datetime.datetime.fromtimestamp(t).replace(microsecond=0)
    t_0 = datetime.datetime.fromtimestamp(t_0).replace(microsecond=0)
    time = t - t_0

    hours, remainder = divmod(time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    strftime = str(hours) + "h:" + str(minutes) + "min:" + str(seconds) + "s"
    return strftime


def adjust_Intensity(image_ref, image, patch, normcrit='linear'):
    """ retruns a normalized version of image_norm to fit the intensity of image_ref"""
    x_1, x_2, y_1, y_2 = patch

    mean_ref_1 = np.mean(np.asarray(image_ref, dtype='float')[y_1:y_2, x_1:x_2])
    mean_im_1 = np.mean(np.asarray(image, dtype='float')[y_1:y_2, x_1:x_2])

    if normcrit == 'linear':
        norm_factor = float(mean_ref_1) / float(mean_im_1)
        # print mean_ref_1, mean_im_1, norm_factor  # debugging
        image_norm = np.asarray(image, dtype='float') * float(norm_factor)
        return image_norm
    elif normcrit == 'shift':
        shift = abs(mean_im_1 - mean_ref_1)
        image_shift = image - shift
        return image_shift


def get_global_minmax(all_Files, dark, ref, normpatch):
    glob_min = 1000000
    glob_max = -1000000

    for i in np.arange(len(all_Files)):
        image_name = all_Files[int(i)]
        image = np.mean(jim.rotate_90(cv2.imread(image_name), 0), axis=-1) - dark
        image = adjust_Intensity(ref, image, normpatch)

        if image.min() < glob_min:
            glob_min = image.min()
        if image.max() > glob_max:
            glob_max = image.max()

    return glob_min, glob_max


def plot_quot(
        all_Files,
        path_to_files,
        gauss=False,
        show_finger_bars=False,
        N=7,
        start=4,
        last=25,
        low=60.,
        high=100.,
        bar_w=10,
        patch=(643, 1779, 1545, 2000),
        normpatch=(1190, 1310, 60, 160),
        normcrit='linear',
        cuts=(0, 0, 0, 0),
        intensities=(0, 0),
        timesteps=(0, 0)
):
    xmin, xmax, ymin, ymax = patch

    fig, axes = plt.subplots(nrows=N, ncols=1)
    left_cut, right_cut, top_cut, bottom_cut = cuts
    dark = np.mean(jim.rotate_90(imop.mean_image(path_to_files, rot=False), 0), axis=-1)
    ref = np.mean(jim.rotate_90(cv2.imread(all_Files[0]), 0), axis=-1) - dark
    # ref = normalize_Intensity(ref, ref, normpatch)

    # first loop: get global min/max values
    print "global stuff"
    glob_min, glob_max = get_global_minmax(all_Files, dark=dark, ref=ref, normpatch=normpatch, start=start, last=last,
                                           N=N)

    # second loop: use global min/max values to create comparable data and plot
    print "There will be a plot soon!"
    t_0 = timesteps[0]
    for ax, i in zip(axes.flat, np.linspace(start, last, num=N)):
        # print N-int(i)  # Countdown
        image_name = all_Files[int(i)]
        image = np.mean(jim.rotate_90(cv2.imread(image_name), 0), axis=-1) - dark
        image = adjust_Intensity(ref, image, normpatch, normcrit=normcrit)

        quot = quotient(image, ref, glob_min, glob_max)
        quot = rescale(quot, low, high)  # viewability

        if gauss:
            quot = ndimage.gaussian_filter(quot, sigma=2.0, order=0)

        # for being sure, imshow does the right thing put in one pixel with 0 and one with 100
        quot[1] = 0.
        quot[2] = 100.

        t = jim.get_timestamp(image_name, human_readable=False)
        title = 't = ' + str(dt2hms(t, t_0))

        im = ax.imshow(quot)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

        if show_finger_bars:
            x1, x2, y1, y2 = patch
            for val, j in zip(intensities[int(i)], np.arange(len(intensities[i]))):
                if val != 0:
                    ax.add_patch(Rectangle((x1 + j - int(bar_w / 2), y1), bar_w, val, color=bar_col, alpha=bar_alph))
        ax.set_xlim([xmin + left_cut, xmax - right_cut])  # works
        ax.set_ylim([ymax - bottom_cut, ymin + top_cut])  # works

    # Make Colorbar
    cbar_ax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cbar_ax, **kw)

    plt.show()


def plot_diff(
        all_Files,
        path_to_files,
        gauss=False,
        show_finger_bars=False,
        N=7,
        start=4,
        last=25,
        low=60.,
        high=100.,
        bar_w=10,
        patch=(643, 1779, 1545, 2000),
        normpatch=(1190, 1310, 60, 160),
        normcrit='linear',
        cuts=(0, 0, 0, 0),
        intensities=(0, 0),
        timesteps=(0, 0)
):
    xmin, xmax, ymin, ymax = patch

    fig, axes = plt.subplots(nrows=N, ncols=1)
    left_cut, right_cut, top_cut, bottom_cut = cuts
    dark = np.mean(jim.rotate_90(imop.mean_image(path_to_files, rot=False), 0), axis=-1)
    ref = np.mean(jim.rotate_90(cv2.imread(all_Files[0]), 0), axis=-1) - dark
    # ref = normalize_Intensity(ref, ref, normpatch)

    # first loop: get global min/max values
    print "global stuff"
    glob_min, glob_max = get_global_minmax(all_Files, dark=dark, ref=ref, normpatch=normpatch, start=start, last=last,
                                           N=N)

    # second loop: use global min/max values to create comparable data and plot
    print "There will be a plot soon!"
    t_0 = timesteps[0]
    for ax, i in zip(axes.flat, np.linspace(start, last, num=N)):
        # print N-int(i)  # Countdown
        image_name = all_Files[int(i)]
        image = np.mean(jim.rotate_90(cv2.imread(image_name), 0), axis=-1) - dark
        image = adjust_Intensity(ref, image, normpatch, normcrit=normcrit)

        diff = difference(image, ref, glob_min, glob_max)
        diff = rescale(diff, low, high)  # viewability

        if gauss:
            diff = ndimage.gaussian_filter(diff, sigma=2.0, order=0)

        # for being sure, imshow does the right thing put in one pixel with 0 and one with 100
        diff[1] = 0.
        diff[2] = 100.

        t = jim.get_timestamp(image_name, human_readable=False)
        title = 't = ' + str(dt2hms(t, t_0))

        im = ax.imshow(diff)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

        if show_finger_bars:
            x1, x2, y1, y2 = patch
            for val, j in zip(intensities[int(i)], np.arange(len(intensities[i]))):
                if val != 0:
                    ax.add_patch(Rectangle((x1 + j - int(bar_w / 2), y1), bar_w, val, color=bar_col, alpha=bar_alph))
        ax.set_xlim([xmin + left_cut, xmax - right_cut])  # works
        ax.set_ylim([ymax - bottom_cut, ymin + top_cut])  # works

    # Make Colorbar
    cbar_ax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cbar_ax, **kw)

    plt.show()


def plot_all(
        all_Files,
        path_to_files,
        savename,
        gauss=False,
        show_finger_bars=False,
        plot_q=True,
        plot_d=True,
        plot_r=True,
        from_file=False,
        globmm_savename='',
        dark_savename='',
        gauss_sigma=2,
        gauss_order=0,
        ref_n=1,
        N=7,
        start=4,
        last=25,
        low_q=60.,
        high_q=100.,
        low_d=0.,
        high_d=100.,
        bar_w=10,
        patch=(643, 1779, 1545, 2000),
        normpatch=(1190, 1310, 60, 160),
        normcrit='linear',
        cuts=(0, 0, 0, 0),
        intensities=(0, 0),
        timesteps=(0, 0)
):
    xmin, xmax, ymin, ymax = patch
    if plot_q and plot_d and plot_r:
        M = 3
    elif plot_q and plot_d and not plot_r:
        M = 2
    elif plot_q and not plot_d and plot_r:
        M = 2
    elif not plot_q and plot_d and plot_r:
        M = 2
    elif not plot_q and not plot_d and plot_r:
        M = 1
    elif not plot_q and plot_d and not plot_r:
        M = 1
    elif plot_q and not plot_d and not plot_r:
        M = 1
    else:
        print "you have to plot something!"
        return 1

    fig, axes = plt.subplots(nrows=N, ncols=M)
    left_cut, right_cut, top_cut, bottom_cut = cuts
    if from_file:
        dark = np.genfromtxt(dark_savename)
    else:
        dark = np.mean(jim.rotate_90(imop.mean_image(path_to_files, rot=False), 0), axis=-1)

    ref = np.mean(jim.rotate_90(cv2.imread(all_Files[ref_n]), 0), axis=-1) - dark
    # ref = normalize_Intensity(ref, ref, normpatch)

    # first loop: get global min/max values
    print "global stuff"
    if from_file:
        glob_min, glob_max = np.genfromtxt(globmm_savename)
    else:
        glob_min, glob_max = get_global_minmax(all_Files, dark=dark, ref=ref, normpatch=normpatch)

    # second loop: use global min/max values to create comparable data and plot
    print "There will be a plot soon!"
    t_0 = timesteps[0]
    mod_3_s0 = np.linspace(0, 3 * N - 3, num=N)
    mod_3_s1 = np.linspace(1, 3 * N - 2, num=N)
    mod_3_s2 = np.linspace(2, 3 * N - 1, num=N)
    mod_2_s0 = np.linspace(0, 2 * N - 2, num=N)
    mod_2_s1 = np.linspace(1, 2 * N - 1, num=N)
    mod_1_s0 = np.linspace(0, 1 * N - 1, num=N)

    i_array = np.reshape(np.arange(M * N), (N, M))
    for i, val in enumerate(np.linspace(start, last, num=N)):
        i_array[i] = int(val)
    i_array = np.reshape(i_array, (M * N,))

    ax_num = 0
    for ax, i in zip(axes.flat, i_array):
        # print N-int(i)  # Countdown
        image_name = all_Files[int(i)]
        raw_image = jim.rotate_90(cv2.imread(image_name), 0)
        image = np.mean(raw_image, axis=-1) - dark
        image = adjust_Intensity(ref, image, normpatch, normcrit=normcrit)

        if plot_d:
            diff = difference(image, ref, glob_min, glob_max)
            diff = rescale(diff, low_d, high_d)  # viewability
        if plot_q:
            quot = quotient(image, ref, glob_min, glob_max)
            quot = rescale(quot, low_q, high_q)  # viewability

        if gauss and plot_d:
            diff = ndimage.gaussian_filter(diff, sigma=2.0, order=0)
        if gauss and plot_q:
            quot = ndimage.gaussian_filter(quot, sigma=2.0, order=0)

        # for being sure, imshow does the right thing put in one pixel with 0 and one with 100
        if plot_d:
            diff[1] = 0.
            diff[2] = 100.
        if plot_q:
            quot[1] = 0.
            quot[2] = 100.

        t = jim.get_timestamp(image_name, human_readable=False)
        title = 't = ' + str(dt2hms(t, t_0))

        if plot_d and plot_q and plot_r:
            if ax_num == 0:
                title = 'Differenzen\nt = ' + str(dt2hms(t, t_0))
            if ax_num == 1:
                title = 'Quotienten\nt = ' + str(dt2hms(t, t_0))
            if ax_num == 2:
                title = 'Rohbild\nt = ' + str(dt2hms(t, t_0))
            if ax_num in mod_3_s0:
                im1 = ax.imshow(diff)
            elif ax_num in mod_3_s1:
                ax.imshow(quot)
            elif ax_num in mod_3_s2:
                ax.imshow(raw_image)

        if plot_d and plot_q and not plot_r:
            if ax_num == 0:
                title = 'Differenzen\nt = ' + str(dt2hms(t, t_0))
            if ax_num == 1:
                title = 'Quotienten\nt = ' + str(dt2hms(t, t_0))
            if ax_num in mod_2_s0:
                im1 = ax.imshow(diff)
            elif ax_num in mod_2_s1:
                ax.imshow(quot)
        if plot_d and not plot_q and plot_r:
            if ax_num == 0:
                title = 'Differenzen\nt = ' + str(dt2hms(t, t_0))
            if ax_num == 1:
                title = 'Rohbild\nt = ' + str(dt2hms(t, t_0))
            if ax_num in mod_2_s0:
                im1 = ax.imshow(diff)
            elif ax_num in mod_2_s1:
                ax.imshow(raw_image)
        if not plot_d and plot_q and plot_r:
            if ax_num == 0:
                title = 'Quotienten\nt = ' + str(dt2hms(t, t_0))
            if ax_num == 1:
                title = 'Rohbild\nt = ' + str(dt2hms(t, t_0))
            if ax_num in mod_2_s0:
                im1 = ax.imshow(quot)
            elif ax_num in mod_2_s1:
                ax.imshow(raw_image)

        if plot_d and not plot_q and not plot_r:
            if ax_num == 0:
                title = 'Differenzen\nt = ' + str(dt2hms(t, t_0))
            if ax_num in mod_1_s0:
                im1 = ax.imshow(diff)
        if not plot_d and plot_q and not plot_r:
            if ax_num == 0:
                title = 'Quotienten\nt = ' + str(dt2hms(t, t_0))
            if ax_num in mod_1_s0:
                im1 = ax.imshow(quot)
        if not plot_d and not plot_q and plot_r:
            if ax_num == 0:
                title = 'Rohbild\nt = ' + str(dt2hms(t, t_0))
            if ax_num in mod_1_s0:
                im1 = ax.imshow(raw_image)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

        if show_finger_bars:
            x1, x2, y1, y2 = patch
            for val, j in zip(intensities[int(i)], np.arange(len(intensities[i]))):
                if val != 0:
                    ax.add_patch(Rectangle((x1 + j - int(bar_w / 2), y1), bar_w, val, color=bar_col, alpha=bar_alph))
        ax.set_xlim([xmin + left_cut, xmax - right_cut])  # works
        ax.set_ylim([ymax - bottom_cut, ymin + top_cut])  # works

        ax_num += 1

    gr = 2
    if M == 1:
        fig.set_size_inches(cm2inch((gr * 21.0, gr * 29.7)))
    else:
        fig.set_size_inches(cm2inch((gr * 29.7, gr * 21.0)))

    # Make Colorbar
    if plot_d or plot_q:
        cbar_ax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im1, cax=cbar_ax, **kw)
    fig.savefig(savename, dpi=300)
    # plt.show()


def main():
    # Load the pictures for plotting purposes
    path_to_files   = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/630_nm'
    path_to_fingers = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/intensities_630.csv'
    path_to_plot    = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/plot_all.pdf'
    globmm_savename = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/glob_mm.csv'
    dark_savename   = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_BCG_150309/measurement_BCG_150309/images/dark.csv'
    all_Files = jim.listdir_nohidden(path_to_files)

    # finger_data = np.genfromtxt(path_to_fingers)
    # timesteps, intensities, pixels, meters = format_data(finger_data)  # [start:end][::step])

    # print all_Files[pic_1]
    # print all_Files[pic_2]

    # plot
    patch = (443, 1500, 1443, 2150)
    normpatch=(1190, 1310,  60, 160)
    xmin, xmax, ymin, ymax = patch

    # Here the real stuff starts!

    # NOT USED
    # plot_quot(
    # all_Files,
    #     path_to_files,
    #     gauss=True,
    #     show_finger_bars=False,
    #     N=7,
    #     start=4,
    #     last=25,
    #     low=60.,
    #     high=100.,
    #     bar_w=10,
    #     patch=(643, 1779, 1545, 2000),
    #     normpatch=(1190, 1310, 60, 160),
    #     normcrit='linear',
    #     cuts=(0, 0, 0, int((ymax - ymin) / 2)),
    #     intensities=intensities,
    #     timesteps=timesteps
    # )

    # NOT USED
    # plot_diff(
    #     all_Files,
    #     path_to_files,
    #     gauss=True,
    #     show_finger_bars=False,
    #     N=7,
    #     start=4,
    #     last=25,
    #     low=0.,
    #     high=100.,
    #     bar_w=10,
    #     patch=(643, 1779, 1545, 2000),
    #     normpatch=(1190, 1310, 60, 160),
    #     normcrit='linear',
    #     cuts=(0, 0, 0, int((ymax - ymin) / 2)),
    #     intensities=intensities,
    #     timesteps=timesteps
    # )


    # # SAVE THE IMPORTANT STUFF
    # ref_n=0
    # dark = np.mean(jim.rotate_90(imop.mean_image(path_to_files, rot=False), 0), axis=-1)
    # np.savetxt(dark_savename, dark, delimiter='\t')
    # ref = np.mean(jim.rotate_90(cv2.imread(all_Files[ref_n]), 0), axis=-1) - dark
    # np.savetxt(globmm_savename, get_global_minmax(all_Files, ref=ref, dark=dark, normpatch=(1190, 1310, 60, 160)), delimiter='\t')

    # used to plot the fingering in the first plot.
    # plot_all(
    #     all_Files,
    #     path_to_files,
    #     savename=path_to_plot,
    #     gauss=True,
    #     show_finger_bars=False,
    #     plot_q=True,
    #     plot_d=False,
    #     plot_r=False,
    #     from_file=True,
    #     dark_savename=dark_savename,
    #     globmm_savename=globmm_savename,
    #     gauss_sigma=1.3,
    #     gauss_order=0,
    #     ref_n=0,
    #     N=10,
    #     start=1,
    #     last=30,
    #     low_d=55.,
    #     high_d=80.,
    #     low_q=67.5,
    #     high_q=75.,
    #     bar_w=10,
    #     patch=(643, 1779, 1545, 2000),
    #     normpatch=(1190, 1310, 60, 160),
    #     normcrit='linear',
    #     cuts=(0, 0, 0, int((ymax - ymin) / 2)),
    #     intensities=intensities,
    #     timesteps=timesteps
    # )


    plot_all(
        all_Files,
        path_to_files,
        savename=path_to_plot,
        gauss=True,
        show_finger_bars=False,
        plot_q=True,
        plot_d=False,
        plot_r=True,
        from_file=True,
        dark_savename=dark_savename,
        globmm_savename=globmm_savename,
        gauss_sigma=2,
        gauss_order=0,
        ref_n=0,
        N=10,
        start=5,
        last=30,
        low_d=55.,
        high_d=80.,
        low_q=67.5,
        high_q=75.,
        bar_w=10,
        patch=patch,
        normpatch=normpatch,
        normcrit='linear',
        cuts=(0, 0, 0, int((ymax - ymin) / 2)),
        intensities='',
        timesteps=''
    )

if __name__ == '__main__':
    main()