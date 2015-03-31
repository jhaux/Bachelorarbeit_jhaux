__author__ = 'jhaux'
# -*- coding: utf8 -*-

import jimlib as jim
import image_operations as imop
import ltm_analysis as ltma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.patches import Rectangle
import scipy.ndimage as ndimage
import numpy as np
import cv2
import datetime

def scalar_cmap(data_points, cmap_name='cool'):
    '''returns an array of rgba colorvalues for plotting purposes'''

    values = np.arange(data_points)
    cmap = cm = plt.get_cmap(cmap_name)
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    color_vals = [scalarMap.to_rgba(values[i]) for i in values]

    return color_vals

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
    """Final Version of the difference method, as described in the Thesis"""
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
    if image.max() == 0:
        pass
    else:
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


def t2hms(t):
    time = datetime.timedelta(t)

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
        image_norm = np.multiply(float(norm_factor), np.asarray(image, dtype='float'))
        print norm_factor * mean_im_1, mean_ref_1, np.mean(image_norm[y_1:y_2, x_1:x_2])
        return image_norm
    elif normcrit == 'shift':
        shift = abs(mean_im_1 - mean_ref_1)
        image_shift = image - shift
        return image_shift
    elif normcrit == 'None':
        return image


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
        absolute_time=False,
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
        bar_col='k',
        bar_alph=0.9,
        patch=(643, 1779, 1545, 2000),
        normpatch=(1190, 1310, 60, 160),
        normcrit='linear',
        cuts=(0, 0, 0, 0),
        fingerlengths=(0, 0),
        timesteps=(0, 0),
        cbar_loc='right',
        x_totalmax=2545, # px
        cell_width=27.3  # cm
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
    t_0 = jim.get_timestamp(all_Files[0], human_readable=False)
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
            diff = ndimage.gaussian_filter(diff, sigma=gauss_sigma, order=gauss_order)
        if gauss and plot_q:
            quot = ndimage.gaussian_filter(quot, sigma=gauss_sigma, order=gauss_order)

        # for being sure, imshow does the right thing put in one pixel with 0 and one with 100
        if plot_d:
            diff[1] = 0.
            diff[2] = 100.
        if plot_q:
            quot[1] = 0.
            quot[2] = 100.

        t = jim.get_timestamp(image_name, human_readable=False)
        if absolute_time:
            title = 't = ' + datetime.datetime.fromtimestamp(t).strftime('%%d.%m.Y %H:%M:%S')
        else:
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

        if M == 1:
            if plot_d:
                if ax_num == 0:
                    title = 'Differenzen\nt = ' + str(dt2hms(t, t_0))
                if ax_num in mod_1_s0:
                    im1 = ax.imshow(diff)
            if plot_q:
                if ax_num == 0:
                    title = 'Quotienten\nt = ' + str(dt2hms(t, t_0))
                if ax_num in mod_1_s0:
                    im1 = ax.imshow(quot)
            if plot_r:
                if ax_num == 0:
                    title = 'Rohbild\nt = ' + str(dt2hms(t, t_0))
                if ax_num in mod_1_s0:
                    im1 = ax.imshow(raw_image)
        ax.set_title(title)

        if ax_num == N-1:
            start_labels = np.zeros(xmin)
            end_labels   = np.zeros(x_totalmax - xmax)
            mid_labels   = np.linspace(0, cell_width, num=(xmax-xmin))
            # total_width = ((xmax - xmin) / cell_width) * x_totalmax
            step =  float(xmax-xmin) / cell_width
            labels = np.hstack([start_labels, end_labels, mid_labels])
            x_array = np.arange(0, x_totalmax, int(5*step))
            print len(x_array), len(labels[::int(5*step)])
            print x_array
            print labels[::int(5*step)]
            ax.set_xticks(x_array)
            ax.set_xticklabels( labels[::int(5*step)])
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        ax.set_yticks([])
        ax.set_yticklabels([])

        if show_finger_bars:
            x1, x2, y1, y2 = patch
            for val, j in zip(fingerlengths[int(i)], np.arange(len(fingerlengths[i]))):
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
        cbar_ax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat], location=cbar_loc)
        plt.colorbar(im1, cax=cbar_ax, **kw)
    fig.savefig(savename, dpi=300, bbox_inches='tight')
    # plt.show()


def plot_patch(
        all_Files,
        path_to_files,
        savename,
        gauss=False,
        show_finger_bars=False,
        plot_q=True,
        plot_d=True,
        plot_r=True,
        plot_adjusted=True,
        from_file=False,
        absolute_time=False,
        name_in_x=True,
        state_type=False,
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
        bar_col='k',
        bar_alph=0.9,
        patch=(643, 1779, 1545, 2000),
        normpatch=(1190, 1310, 60, 160),
        normcrit='linear',
        cuts=(0, 0, 0, 0),
        fingerlengths=(0, 0),
        timesteps=(0, 0),
        make_cbar=True,
        cbar_loc='right',
        x_totalmax=2545.,
        cell_width=27.3
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

    fig, axes = plt.subplots(nrows=M, ncols=N)
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
    t_0 = jim.get_timestamp(all_Files[0], human_readable=False)
    mod_3_s0 = np.linspace(0 * 0, 1 * N-1, num=N)
    mod_3_s1 = np.linspace(1 * N, 2 * N-1, num=N)
    mod_3_s2 = np.linspace(2 * N, 3 * N-1, num=N)
    mod_2_s0 = np.linspace(0 * 0, 1 * N-1, num=N)
    mod_2_s1 = np.linspace(1 * N, 2 * N-1, num=N)
    mod_1_s0 = np.linspace(0 * 0, 1 * N-1, num=N)

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
            diff = ndimage.gaussian_filter(diff, sigma=gauss_sigma, order=gauss_order)
        if gauss and plot_q:
            quot = ndimage.gaussian_filter(quot, sigma=gauss_sigma, order=gauss_order)

        # for being sure, imshow does the right thing put in one pixel with 0 and one with 100
        if plot_d:
            diff[1] = 0.
            diff[2] = 100.
        if plot_q:
            quot[1] = 0.
            quot[2] = 100.

        if plot_adjusted:
            raw_image = image

        t = jim.get_timestamp(image_name, human_readable=False)
        if absolute_time:
            title = 't = ' + datetime.datetime.fromtimestamp(t).strftime('%%d.%m.Y %H:%M:%S')
        else:
            title = 't = ' + str(dt2hms(t, t_0))

        if plot_d and plot_q and plot_r:
            if state_type:
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
            if state_type:
                if ax_num == 0:
                    title = 'Differenzen\nt = ' + str(dt2hms(t, t_0))
                if ax_num == 1:
                    title = 'Quotienten\nt = ' + str(dt2hms(t, t_0))
            if ax_num in mod_2_s0:
                im1 = ax.imshow(diff)
            elif ax_num in mod_2_s1:
                ax.imshow(quot)
        if plot_d and not plot_q and plot_r:
            if state_type:
                if ax_num == 0:
                    title = 'Differenzen\nt = ' + str(dt2hms(t, t_0))
                if ax_num == 1:
                    title = 'Rohbild\nt = ' + str(dt2hms(t, t_0))
            if ax_num in mod_2_s0:
                im1 = ax.imshow(diff)
            elif ax_num in mod_2_s1:
                ax.imshow(raw_image)
        if not plot_d and plot_q and plot_r:
            if state_type:
                if ax_num == 0:
                    title = 'Quotienten\nt = ' + str(dt2hms(t, t_0))
                if ax_num == 1:
                    title = 'Rohbild\nt = ' + str(dt2hms(t, t_0))
            if ax_num in mod_2_s0:
                im1 = ax.imshow(quot)
            elif ax_num in mod_2_s1:
                ax.imshow(raw_image)

        if M == 1:
            if plot_d:
                if state_type:
                    if ax_num == 0:
                        title = 'Differenzen\nt = ' + str(dt2hms(t, t_0))
                if ax_num in mod_1_s0:
                    im1 = ax.imshow(diff)
            if plot_q:
                if state_type:
                    if ax_num == 0:
                        title = 'Quotienten\nt = ' + str(dt2hms(t, t_0))
                if ax_num in mod_1_s0:
                    im1 = ax.imshow(quot)
            if plot_r:
                if state_type:
                    if ax_num == 0:
                        title = 'Rohbild\nt = ' + str(dt2hms(t, t_0))
                if ax_num in mod_1_s0:
                    im1 = ax.imshow(raw_image)

        if name_in_x:
            ax.set_xlabel(title, rotation=45)
        else:
            ax.set_title(title)

        if ax_num == N:
            start_labels = np.zeros(xmin)
            end_labels   = np.zeros(x_totalmax - xmax)
            mid_labels   = np.linspace(0, 27.3, num=(xmax-xmin))
            labels = np.hstack([start_labels, end_labels, mid_labels])
            ax.set_xticklabels(labels)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        ax.set_yticks([])
        ax.set_yticklabels([])

        if show_finger_bars:
            x1, x2, y1, y2 = patch
            for val, j in zip(fingerlengths[int(i)], np.arange(len(fingerlengths[i]))):
                if val != 0:
                    ax.add_patch(Rectangle((x1 + j - int(bar_w / 2), y1), bar_w, val, color=bar_col, alpha=bar_alph))
        ax.set_xlim([xmin + left_cut, xmax - right_cut])  # works
        ax.set_ylim([ymax - bottom_cut, ymin + top_cut])  # works

        ax_num += 1

    gr = 2
    if M == 1:
        fig.set_size_inches(cm2inch((gr * 29.7, gr * 21.0)))
    else:
        fig.set_size_inches(cm2inch((gr * 21.0, gr * 29.7)))

    # Make Colorbar
    if make_cbar:
        if plot_d or plot_q:
            cbar_ax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat], location=cbar_loc)
            plt.colorbar(im1, cax=cbar_ax, **kw)
    fig.savefig(savename, dpi=300, bbox_inches='tight')
    # plt.show()


def finger_count(intensities):
    '''Take an array of intensities and look at the first and second derivative. Assuning that at every maximum there
    is a finger the first derivative needs to be zero and the second negative. Store an array of the same dimension with
    a 1 at the location of the finger and a 0 everywhere there is not. Additionally return the count of fingers.'''

    first_derivative  = np.diff(intensities)
    second_derivative = np.diff(intensities, n=2)

    result_array = np.zeros(intensities.shape)
    # result_array[ 0] = 0 # compensate, that you cannot look at all pixels
    # result_array[-1] = 0
    # result_array[-2] = 0

    n_fingers = 0
    # iterate over every pixel. -2 because len(sec_deriv) = len(intens) - 2. Start at 1, because the first pixel has no difference to the one before.
    for pixel in np.linspace(1,len(intensities)-2, num=len(intensities)-3):
        if np.diff(np.sign(first_derivative))[pixel-1] < 0 and np.sign(second_derivative)[pixel-1] == -1:
            result_array[pixel] = 1
            n_fingers += 1
        # else:
        #     result_array[pixel] = 0

    return result_array, n_fingers

def fclean_intensities(intensities, handle=25000, cell_width=0.23, clean_criterium='amplitude'):
    '''all_data must be a one-column array!'''

    N = intensities.shape[0]

    waves = intensities

    # map real spacial coordinates to the pixel data:
    xmin = 0.   # [m]
    xmax = cell_width # [m]
    xlength = xmax - xmin

    # get wavelengthspace and corresponding wavelngths
    wavelengthspace  = np.fft.rfft(waves)
    wavelengths      = np.fft.rfftfreq(waves.shape[0], d=xlength/waves.shape[0]) # d: Distance of datapoints in "Space-space"

    # clean wavlengthspace
    wavelengthspace_clean = np.empty_like(wavelengthspace)
    wavelengthspace_clean[:] = wavelengthspace
    if clean_criterium == 'amplitude':
        wavelengthspace_clean[(abs(wavelengthspace) < handle)] = 0  # filter all unwanted wavelengths
    elif clean_criterium == 'wavelength':
        wavelengthspace_clean[(wavelengths > handle)] = 0

    # get cleaned version of the waves
    waves_clean   = np.fft.irfft(wavelengthspace_clean)  # inverse fft returns cleaned wave
    x_array_clean = np.linspace(xmin, xmax, num=waves_clean.shape[0])

    return waves_clean, x_array_clean

def get_wavelengthspace(intensities, handle=25000, cell_width=0.23, clean_criterium='amplitude'):


    N = intensities.shape[0]

    waves = intensities

    # map real spacial coordinates to the pixel data:
    xmin = 0.   # [m]
    xmax = cell_width # [m]
    xlength = xmax - xmin

    # get wavelengthspace and corresponding wavelngths
    wavelengthspace  = np.fft.rfft(waves)
    wavelengths      = np.fft.rfftfreq(waves.shape[0], d=xlength/waves.shape[0]) # d: Distance of datapoints in "Space-space"

    return wavelengthspace, wavelengths


def finger_length(image, patch, threshold, handle=150, clean_criterium='wavelength', cell_width=0.23):
    '''Find out where there are fingers and how long they are by looking at the image-patch of interest, applying the
    finger_count function (where) and then look from down to top at those positions, where the first pixel greater
    than the threshold appears.'''

    intensities = ltma.get_raw_intensities(image, patch)
    # get rid of the noise
    intensities_clean, meters_clean = fclean_intensities(intensities, handle=handle, clean_criterium=clean_criterium, cell_width=cell_width)
    # get the positions of the fingers
    finger_positions, N_fingers = finger_count(intensities_clean)

    # make a binery image out of the patch of the image:
    x_1, x_2, y_1, y_2 = patch
    image_patch = image[y_1:y_2,x_1:x_2]
    ydim, xdim = image_patch.shape

    # walk from top to bottom to get the fingerlength. To avoid errors take the mean of nine neighbouring pixels
    # first for: walk through columns. If there is a finger take a measurement.
    for position in np.arange(4,len(finger_positions)-5):
        if finger_positions[position] == 0:
            pass
        else:
            #second for: walk through pixels in one column
            for pixel in np.arange(ydim)[::-1]: #::-1 => look at the column from down under
                if np.mean(image_patch[pixel,position-4:position+5]) > threshold:
                    finger_positions[position] = pixel + 1 # +1 because the first counts as 0
                    break

    return finger_positions


def save_finger_lengths(doutfile, qoutfile, all_Files, dark_savename, globmm_savename,
                        patch, ref_n, normpatch, normcrit='linear',
                        low_q=60., high_q=100., low_d=0., high_d=100.,
                        from_file=True, save_d=True, save_q=True,
                        q_threshold=0.5, d_threshold=0.5, handle=150, clean_criterium='wavelength', cell_width=0.23):


    xmin, xmax, ymin, ymax = patch

    if from_file:
        dark = np.genfromtxt(dark_savename)
    else:
        dark = np.mean(jim.rotate_90(imop.mean_image(path_to_files, rot=False), 0), axis=-1)

    ref = np.mean(jim.rotate_90(cv2.imread(all_Files[ref_n]), 0), axis=-1) - dark

    # first loop: get global min/max values
    print "global stuff"
    if from_file:
        glob_min, glob_max = np.genfromtxt(globmm_savename)
    else:
        glob_min, glob_max = get_global_minmax(all_Files, dark=dark, ref=ref, normpatch=normpatch)

    # second loop: use global min/max values to create comparable data
    t_0 = jim.get_timestamp(all_Files[0], human_readable=False)
    if save_d:
        # first empty the file
        open(doutfile, 'w').close()
        # then write into it line per line

        image_name = all_Files[int(0)]
        raw_image = jim.rotate_90(cv2.imread(image_name), 0)
        image = np.mean(raw_image, axis=-1) - dark
        image = adjust_Intensity(ref, image, normpatch, normcrit=normcrit)

        diff = difference(image, ref, glob_min, glob_max)
        diff = rescale(diff, low_d, high_d)  # viewability

        t = jim.get_timestamp(image_name, human_readable=False)
        dt = t -t_0

        finger_lengths = finger_length(image=diff, patch=patch, threshold=d_threshold, handle=handle, clean_criterium=clean_criterium, cell_width=cell_width)

        temp_array = np.append(dt, finger_lengths)
        d_save_array = temp_array

        for i in np.arange(len(all_Files)):
            image_name = all_Files[int(i)]
            raw_image = jim.rotate_90(cv2.imread(image_name), 0)
            image = np.mean(raw_image, axis=-1) - dark
            image = adjust_Intensity(ref, image, normpatch, normcrit=normcrit)

            diff = difference(image, ref, glob_min, glob_max)
            diff = rescale(diff, low_d, high_d)  # viewability

            t = jim.get_timestamp(image_name, human_readable=False)
            dt = t -t_0

            finger_lengths = finger_length(image=diff, patch=patch, threshold=d_threshold, handle=handle, clean_criterium=clean_criterium, cell_width=cell_width)

            temp_array = np.append(dt, finger_lengths)
            d_save_array = np.vstack([d_save_array, temp_array])

        with open(doutfile, 'a') as f_handle:
            np.savetxt(f_handle, d_save_array, delimiter='\t')

    if save_q:
        # first empty the file:
        open(qoutfile, 'w').close()
        # then write into it line per line
        # do image operations
        image_name = all_Files[int(i)]
        raw_image = jim.rotate_90(cv2.imread(image_name), 0)
        image = np.mean(raw_image, axis=-1) - dark
        image = adjust_Intensity(ref, image, normpatch, normcrit=normcrit)

        # get quotient data
        quot = quotient(image, ref, glob_min, glob_max)
        quot = rescale(quot, low_q, high_q)  # viewability

        # get duration at timestep
        t = jim.get_timestamp(image_name, human_readable=False)
        dt = t - t_0

        finger_lengths = finger_length(image=quot, patch=patch, threshold=q_threshold, handle=handle, clean_criterium=clean_criterium, cell_width=cell_width)

        temp_array = np.append(dt, finger_lengths)
        q_save_array = temp_array

        for i in np.arange(len(all_Files)):
            # do image operations
            image_name = all_Files[int(i)]
            raw_image = jim.rotate_90(cv2.imread(image_name), 0)
            image = np.mean(raw_image, axis=-1) - dark
            image = adjust_Intensity(ref, image, normpatch, normcrit=normcrit)

            # get quotient data
            quot = quotient(image, ref, glob_min, glob_max)
            quot = rescale(quot, low_q, high_q)  # viewability

            # get duration at timestep
            t = jim.get_timestamp(image_name, human_readable=False)
            dt = t - t_0

            finger_lengths = finger_length(image=quot, patch=patch, threshold=q_threshold, handle=handle, clean_criterium=clean_criterium, cell_width=cell_width)

            temp_array = np.append(dt, finger_lengths)
            q_save_array = np.vstack([q_save_array, temp_array])

        with open(qoutfile, 'a') as f_handle:
            np.savetxt(f_handle, q_save_array, delimiter='\t')

    return 0


def save_raw_intensities(
        all_Files,
        path_to_files,
        doutfile,
        qoutfile,
        save_q=True,
        save_d=True,
        from_file=True,
        globmm_savename='',
        dark_savename='',
        ref_n=0,
        low_q=60.,
        high_q=100.,
        low_d=0.,
        high_d=100.,
        patch=(643, 1779, 1545, 2000),
        normpatch=(1190, 1310, 60, 160),
        normcrit='linear',
        cuts=(0, 0, 0, 0)
):

    xmin, xmax, ymin, ymax = patch
    left_cut, right_cut, top_cut, bottom_cut = cuts

    # cut patch to desired size
    xmin, xmax, ymin, ymax = xmin + left_cut, xmax - right_cut, ymin + top_cut, ymax - bottom_cut

    if from_file:
        dark = np.genfromtxt(dark_savename)
    else:
        dark = np.mean(jim.rotate_90(imop.mean_image(path_to_files, rot=False), 0), axis=-1)

    ref = np.mean(jim.rotate_90(cv2.imread(all_Files[ref_n]), 0), axis=-1) - dark

    # first loop: get global min/max values
    print "global stuff"
    if from_file:
        glob_min, glob_max = np.genfromtxt(globmm_savename)
    else:
        glob_min, glob_max = get_global_minmax(all_Files, dark=dark, ref=ref, normpatch=normpatch)

    # second loop: use global min/max values to create comparable data
    t_0 = jim.get_timestamp(all_Files[0], human_readable=False)
    if save_d:
        # first empty the file
        open(doutfile, 'w').close()
        # then write into it line per line
        save_array = np.ndarray(shape=(1, xmax-xmin + 1))
        for i in np.arange(len(all_Files)):
            image_name = all_Files[int(i)]
            raw_image = jim.rotate_90(cv2.imread(image_name), 0)
            image = np.mean(raw_image, axis=-1) - dark
            image = adjust_Intensity(ref, image, normpatch, normcrit=normcrit)

            diff = difference(image, ref, glob_min, glob_max)
            diff = rescale(diff, low_d, high_d)  # viewability

            t = jim.get_timestamp(image_name, human_readable=False)
            dt = t -t_0

            diff_patch = diff[ymin:ymax, xmin:xmax]
            int_diff = np.mean(diff_patch, axis=0)

            temp_array = np.append(dt, int_diff)
            save_array = np.vstack([save_array, temp_array])

        with open(doutfile, 'a') as f_handle:
            np.savetxt(f_handle, save_array, delimiter='\t')


    if save_q:
        # first empty the file
        open(qoutfile, 'w').close()
        # then write into it line per line
        save_array = np.ndarray(shape=(1, xmax-xmin + 1))
        for i in np.arange(len(all_Files)):
            # do image operations
            image_name = all_Files[int(i)]
            raw_image = jim.rotate_90(cv2.imread(image_name), 0)
            image = np.mean(raw_image, axis=-1) - dark
            image = adjust_Intensity(ref, image, normpatch, normcrit=normcrit)

            # get quotient data
            quot = quotient(image, ref, glob_min, glob_max)
            quot = rescale(quot, low_q, high_q)  # viewability

            # get duration at timestep
            t = jim.get_timestamp(image_name, human_readable=False)
            dt = t -t_0

            # get mean intensities
            quot_patch = quot[ymin:ymax, xmin:xmax]
            int_quot = np.mean(quot_patch, axis=0)

            # save as txt
            temp_array = np.append(dt, int_quot)
            save_array = np.vstack([save_array, temp_array])

        with open(qoutfile, 'a') as f_handle:
            np.savetxt(f_handle, save_array, delimiter='\t')

    return 0

def plot_fingerdrift(savename, all_Files, patch, normpatch,
                     globmm_savename, dark_savename,
                     line=0, ref_n=0, normcrit='linear',
                     start=0, stop=None,
                     low_q=0, high_q=100,
                     from_file=True, gauss=True,
                     gauss_sigma=2, gauss_order=0,
                     figsize=(10,10), add_length=5):


    xmin, xmax, ymin, ymax = patch

    if from_file:
        dark = np.genfromtxt(dark_savename)
    else:
        dark = np.mean(jim.rotate_90(imop.mean_image(path_to_files, rot=False), 0), axis=-1)

    ref = np.mean(jim.rotate_90(cv2.imread(all_Files[ref_n]), 0), axis=-1) - dark

    # first loop: get global min/max values
    print "global stuff"
    if from_file:
        glob_min, glob_max = np.genfromtxt(globmm_savename)
    else:
        glob_min, glob_max = get_global_minmax(all_Files, dark=dark, ref=ref, normpatch=normpatch)

    # second loop: use global min/max values to create plot array
    t_0 = jim.get_timestamp(all_Files[0], human_readable=False)
    plotarray = np.ndarray(shape=(1, xmax-xmin))
    times = np.array([])
    for i in np.arange(len(all_Files))[start:stop]:
        # do image operations
        image_name = all_Files[int(i)]
        raw_image = jim.rotate_90(cv2.imread(image_name), 0)
        image = np.mean(raw_image, axis=-1) - dark
        image = adjust_Intensity(ref, image, normpatch, normcrit=normcrit)

        # get quotient data
        quot = quotient(image, ref, glob_min, glob_max)
        quot = rescale(quot, low_q, high_q)  # viewability

        # get duration at timestep
        t = jim.get_timestamp(image_name, human_readable=False)

        # get mean intensities
        quot_patch = quot[ymin+line:ymin+line+1, xmin:xmax]
        for l in np.arange(add_length):
            plotarray = np.vstack([plotarray, quot_patch])
            np.append(times, t)
            l += 1

    if gauss:
        plotarray = ndimage.gaussian_filter(plotarray, sigma=gauss_sigma, order=gauss_order)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(plotarray)
    ax.set_yticklabels([dt2hms(t, t_0) for t in times])
    ax.set_ylabel('time')

    cbar_ax, kw = mpl.colorbar.make_axes([ax])
    plt.colorbar(im, cax=cbar_ax, **kw)

    fig.savefig(savename, dpi=300)

    return 0

def save_darkframe(dark_savename, path_to_files):
    dark = np.mean(jim.rotate_90(imop.mean_image(path_to_files, rot=False), 0), axis=-1)
    np.savetxt(dark_savename, dark, delimiter='\t')

def save_glob_minmax(globmm_savename, all_Files, dark, normpatch, ):
    ref = np.mean(jim.rotate_90(cv2.imread(all_Files[0]), 0), axis=-1) - dark
    np.savetxt(globmm_savename, get_global_minmax(all_Files, ref=ref, dark=dark, normpatch=normpatch), delimiter='\t')

def plot_fft(savename, all_intensities, start=0, stop=None, step=6,
             handle=150, clean_criterium='wavelength', cell_width=0.23,
             cmap_name='jet', xlims=(0, 0.23), ylims=(0, 100), alpha=0.5, figsize=(1,1)):

    N_datapoints = len(all_intensities[start:stop][::step])
    colors = scalar_cmap(N_datapoints, cmap_name)
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)
    for i, c_i in zip(np.arange(len(all_intensities))[start:stop][::step][::-1], np.arange(N_datapoints)[::-1]):
        wavelengths, x_array = fclean_intensities(intensities=all_intensities[i], handle=handle, clean_criterium=clean_criterium, cell_width=cell_width)
        ax.plot(x_array, wavelengths, c=colors[c_i], linewidth=0.5, alpha=alpha)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel('horizontale Position in der Zelle $[cm]$')
    ax.set_ylabel(u'Durchschnittsintensität')
    fig.savefig(savename, dpi=300, bbox_inches='tight')
    return 0

def plot_wavelengthspace(savename, all_intensities, start=0, stop=None, step=6,
                         handle=150, clean_criterium='wavelength', cell_width=0.23,
                         cmap_name='jet', xlims=(0, 3), ylims=(0, 3000), alpha=0.5, figsize=(10,10)):

    N_datapoints = len(all_intensities[start:stop][::step])
    colors = scalar_cmap(N_datapoints, cmap_name)
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)
    for i, c_i in zip(np.arange(len(all_intensities))[start:stop][::step][::-1], np.arange(N_datapoints)[::-1]):
        wavelengths, x_array = get_wavelengthspace(intensities=all_intensities[i], handle=handle, clean_criterium=clean_criterium, cell_width=cell_width)
        ax.plot(x_array, abs(wavelengths), c=colors[c_i], linewidth=0.5, alpha=alpha)
        if c_i == N_datapoints - 1:
            a_max = max(abs(wavelengths[(x_array < handle) & (x_array > 0.2*handle)]))
            k_max = x_array[abs(wavelengths) == a_max][0]

            print 0.2*handle, handle
            print max(abs(wavelengths[(x_array < handle) & (x_array > 0.2*handle)]))
            print k_max
    ax.plot((k_max,k_max),(-1,100000000000), 'k--', label='$k_{max} = ' + str(k_max)[0:5] + 'cm^{-1}$')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel('$k \, [cm^{-1}]$')
    ax.set_ylabel('Amplitude $\left|A\\right|$')
    plt.legend()
    fig.savefig(savename, dpi=300, bbox_inches='tight')
    return 0

def plot_fingercount(savename, fingers, times):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_array = []
    for f in fingers:
        N_fingers = 0
        for vals in f:
            if vals > 0:
                N_fingers += 1
        n_array.append(N_fingers)

    # times = [t2hms(t) for t in times]
    ax.plot(n_array)

    ax.set_xlim((5, 50))
    t_list = []
    times = np.linspace(9*60, 5059, num=10)
    for t in times:
        hours, remainder = divmod(t, 3600)
        minutes, seconds = divmod(remainder, 60)
        if int(seconds) > 9:
            strftime = str(int(hours)) + ":" + str(int(minutes))+":" + str(int(seconds))
        else:
            strftime = str(int(hours)) + ":" + str(int(minutes))+":0" + str(int(seconds))
        t_list.append(strftime)

    ax.set_xticklabels(t_list, rotation=45)
    fig.subplots_adjust(bottom=0.15)
    ax.set_ylim((0, 75))
    ax.set_xlabel('Zeit')
    ax.set_ylabel('Anzahl detektierter Finger')

    fig.savefig(savename, dpi=300)

def plot_fingergrowth(savename, fingers, times, cell_width=27.3, x_length=100):
    fig = plt.figure()
    ax = fig.add_subplot(111)


    cm2px = cell_width / x_length

    l_array = []
    for f in fingers:
        l_fingers = 0
        n_fingers = 0
        for vals in f:
            if vals > 0:
                l_fingers += vals
                n_fingers += 1.
        if n_fingers == 0:
            n_fingers = 1
        l_array.append(cm2px * (l_fingers / n_fingers))

    ax.plot(l_array)

    ax.set_xlim((5, 50))
    t_list = []
    times = np.linspace(9*60, 5059, num=10)
    for t in times:
        hours, remainder = divmod(t, 3600)
        minutes, seconds = divmod(remainder, 60)
        if int(seconds) > 9:
            strftime = str(int(hours)) + ":" + str(int(minutes))+":" + str(int(seconds))
        else:
            strftime = str(int(hours)) + ":" + str(int(minutes))+":0" + str(int(seconds))
        t_list.append(strftime)

    ax.set_xticklabels(t_list, rotation=45)
    fig.subplots_adjust(bottom=0.15)
    ax.set_xlabel('Zeit')
    ax.set_ylabel(u'Durchschnittliche Länge der Finger $[cm]$')
    ax.set_title('Durchschnittliches Wachstum der Finger')

    fig.savefig(savename, dpi=300)

def plot_fingergrowth_patch(savename, xpatch, fingers, times, cell_width=27.3, x_length=100):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xmin, xmax = xpatch

    cm2px = cell_width / x_length

    l_array = []
    for f in fingers:
        l_fingers = 0
        n_fingers = 0
        for vals in f[xmin:xmax]:
            if vals > 0:
                l_fingers += vals
                n_fingers += 1.
        if n_fingers == 0:
            n_fingers = 1
        l_array.append(cm2px * (l_fingers / n_fingers))

    ax.plot(l_array)

    t_list = []
    times = np.linspace(0, 18360, num=len(fingers)+1)
    for t in times:
        hours, remainder = divmod(t, 3600)
        minutes, seconds = divmod(remainder, 60)
        strftime = str(int(minutes))#+":0" + str(int(seconds))
        t_list.append(strftime)
    ax.set_xticks((np.arange(len(fingers))))
    ax.set_xticklabels(t_list, rotation=45)
    ax.set_xlim((5, 23))
    fig.subplots_adjust(bottom=0.15)
    ax.set_xlabel('Zeit')
    ax.set_ylabel(u'Länge des Fingers $[cm]$')
    ax.set_title('Wachstum eines einzelnen Fingers')

    fig.savefig(savename, dpi=300)

def plot_fingergrowth_single(savename, fingers, all_Files, p_patch, rects, ss=(0, None), cell_width=27.3, x_length=100):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    image = jim.rotate_90(cv2.imread(all_Files[10]), 0)
    # xmin, xmax = xpatch
    xp_min, xp_max, yp_min, yp_max = p_patch
    start, stop = ss
    # take only what is needed from fingers:
    n_array = []
    stepsize = 10
    big_mama = 0

    for i, f in enumerate(fingers):

        rest =  len(f) % stepsize
        f = f[:-rest]
        ls = np.mean(f.reshape(-1, stepsize), axis=-1)
        all_f = len(f[f > 0])

        # print ls

        if big_mama == 0:
            l_array = ls
        l_array = np.vstack([l_array, ls])
        n_array.append(all_f)

    l_array = np.rot90(l_array)

    ax1.plot(n_array)
    ax1.set_title('start = ' + jim.get_timestamp(all_Files[0]))
    ax1.set_xticklabels([jim.get_timestamp(t) for t in all_Files[start:stop]], rotation=45)

    colors = scalar_cmap(len(l_array))
    for i, l in enumerate(l_array):
        ax2.plot(l, c=colors[i])

    ax2.set_xticklabels([jim.get_timestamp(t) for t in all_Files[start:stop]], rotation=45)
    # t_list = []
    # times = np.linspace(0, 18360, num=len(fingers)+1)
    # for t in times:
    #     hours, remainder = divmod(t, 3600)
    #     minutes, seconds = divmod(remainder, 60)
    #     strftime = str(int(minutes))#+":0" + str(int(seconds))
    #     t_list.append(strftime)
    # ax.set_xticks((np.arange(len(fingers))))
    # ax.set_xticklabels(t_list, rotation=45)
    # ax.set_xlim((5, 23))
    # fig.subplots_adjust(bottom=0.15)
    # ax.set_xlabel('Zeit')
    # ax.set_ylabel(u'Länge des Fingers $[cm]$')
    # ax.set_title('Wachstum eines einzelnen Fingers')

    # fig.savefig(savename, dpi=300)
    plt.show()

def main():
    # Load the pictures for plotting purposes
    path_to_files = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/630_nm'
    path_to_fingers = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/intensities_630.csv'
    path_to_plot = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/plot_all.pdf'
    globmm_savename = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/glob_mm.csv'
    dark_savename = u'/Users/jhaux/Desktop/Bachelorarbeit/Measurements/measurement_2015-02-02_14-03-19/measurement_2015-02-02_14-03-19/images/dark.csv'
    all_Files = jim.listdir_nohidden(path_to_files)

    finger_data = np.genfromtxt(path_to_fingers)
    timesteps, intensities, pixels, meters = format_data(finger_data)  # [start:end][::step])

    # print all_Files[pic_1]
    # print all_Files[pic_2]

    # plot
    patch = (643, 1779, 1545, 2000)
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


    # SAVE THE IMPORTANT STUFF
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
        patch=(643, 1779, 1545, 2000),
        normpatch=(1190, 1310, 60, 160),
        normcrit='linear',
        cuts=(0, 0, 0, int((ymax - ymin) / 2)),
        intensities=intensities,
        timesteps=timesteps
    )

if __name__ == '__main__':
    main()