from bokeh.models.markers import X
from bokeh.plotting import ColumnDataSource, figure
from bokeh.models import Label, HoverTool
from bokeh.plotting import output_notebook, show, figure
from bokeh.layouts import gridplot, column, row
from bokeh.models.glyphs import Line
from bokeh.models import Label, HoverTool, Patch
from scipy import stats
from collections import Counter
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib
from .sampletype_binary import sampletype_binary
from .calc_rsd_dratio_blank import calc_rsd_dratio_blank
from .table_check import table_check


def dist_plot(DataTable, PeakTable, parametric=True, batch='all', plot='all', colormap='Accent', scale_x=1, scale_y=1, padding=0.05, smooth=None, alpha=0.05, legend=True):

    DataTable = sampletype_binary(DataTable)  # Create binary columns for ['QC', 'QCT', 'Sample', 'Blank']

    if batch == 'all':
        batch = list(DataTable.Batch.unique())  # all batches

    if type(batch) == int:
        batch = [batch]  # put batch in a list

    if plot == 'all':
        plot = list(DataTable.SampleType.unique())  # all samples

    # Error check: DataTable
    table_check(DataTable, print_statement=False)

    # Error check: batch
    batch_all = list(DataTable.Batch.unique())
    for i in batch:
        if i not in batch_all:
            print("{} is not a batch in the DataTable".format(i))
            return

    batch_count = Counter(batch)
    batch_duplicates = [k for k, v in batch_count.items() if v > 1]
    if len(batch_duplicates) > 0:
        print("There are duplicates in batch: {}".format(batch_duplicates))
        return

    # Error check: parametric
    parametric_check = [True, False]
    if parametric not in parametric_check:
        print("parametric must be True, or False. {} is not an option".format(parametric))
        return

    # Error check: legend
    legend_check = [True, False]
    if legend not in legend_check:
        print("legend must be True, or False. {} is not an option".format(parametric))
        return

    # Error check: plot
    plot_check = list(DataTable.SampleType.unique())
    for i in plot:
        if i not in plot_check:
            print("plot can only include the following: {}. {} is not an option".format(plot_check, i))
            return

    # Error check scale
    if scale_x < 1:
        print("Please use a value greater or equal to 1 for scale_x")
        return

    if scale_y < 1:
        print("Please use a value greater or equal to 1 for scale_y")
        return

    # Alpha
    if alpha < 0:
        print("alpha must be between 0 and 1")
        return
    elif alpha > 1:
        print("alpha must be between 0 and 1")
        return

    # padding
    if padding < 0:
        print("padding must be at least 0")
        return

    # smooth
    if smooth == None:
        pass
    else:
        if smooth < 0:
            print("smooth must be at least 0 or None")
            return

    # Error check: colormap
    colormap_all = "Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r"
    colormap_check = [i.strip() for i in colormap_all.split(',')]
    if colormap not in colormap_check:
        print("{} is not an option for colormap. Possible options include 'Set1', 'Set2', 'rainbow'. All options for colormap can be found here: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html".format(colormap))
        return

    # Plot
    if len(batch) == 1:
        dist_plot_singlebatch(DataTable, PeakTable, parametric=parametric, batch=batch, plot=plot, colormap=colormap, scale_x=scale_x, scale_y=scale_y, padding=padding, smooth=smooth, alpha=alpha, legend=legend)
    else:
        dist_plot_multibatch(DataTable, PeakTable, parametric=parametric, batch=batch, plot=plot, colormap=colormap, scale_x=scale_x, scale_y=scale_y, padding=padding, smooth=smooth, alpha=alpha, legend=legend)


def dist_plot_singlebatch(DataTable, PeakTable, parametric=True, batch='all', plot='all', colormap='Accent', scale_x=1, scale_y=1, padding=0.05, smooth=None, alpha=0.05, legend=True):

    # Get BatchTable
    batch_member = np.where(DataTable.Batch.isin(batch), 1, 0)  # only batch
    BatchTable = DataTable[batch_member == 1]

    # Extract important info
    peak_list = PeakTable.Name
    x = BatchTable[peak_list].values

    # Set width and height
    width = 600 * scale_x
    height = 500 * scale_y

    # Plot figure
    fig = figure(title="RSD (Batch {})".format(batch[0]), x_axis_label='RSD', y_axis_label="Density", plot_height=height, plot_width=width)
    fig.ygrid.visible = False
    fig.xgrid.visible = False
    fig.title.text_font_size = '16pt'
    fig.xaxis.axis_label_text_font_size = "12pt"
    fig.yaxis.axis_label_text_font_size = "12pt"

    for i in plot:
        x_plot = x[BatchTable.SampleType == i]  # Get X

        # Get color
        if i == 'Blank':
            col = '#00FF00'
        elif i == 'Sample':
            colmap = plt.get_cmap(colormap)
            batch_i = batch[0]
            b_rgb = colmap([batch_i])
            b_hex = matplotlib.colors.rgb2hex(b_rgb[0])
            col = b_hex
        elif i == 'QC':
            col = '#FF0000'
        elif i == 'QCW':
            col = '#FFC000'
        elif i == 'QCB':
            col = '#FFFC00'
        elif i == 'QCT':
            col = '#00FFFF'
        else:
            pass

        # add dist to plot
        add_dist_to_plot(fig, x_plot, sampletype=i, col=col, parametric=parametric, padding=padding, smooth=smooth, alpha=alpha)

    # legend
    if legend == False:
        fig.legend.visible = False

    output_notebook()
    show(fig)


def dist_plot_multibatch(DataTable, PeakTable, parametric=True, batch='all', plot='all', colormap='Accent', scale_x=1, scale_y=1, padding=0.05, smooth=None, alpha=0.05, legend=True):

    # Get BatchTable
    batch_member = np.where(DataTable.Batch.isin(batch), 1, 0)  # only batch
    BatchTable = DataTable[batch_member == 1]

    # Extract important info
    peak_list = PeakTable.Name
    x = BatchTable[peak_list].values

    # Set width and height
    width = 500 * scale_x
    height = 500 * scale_y

    # Plot figure
    grid = []

    # Overall fig
    fig = figure(title="RSD (All)", x_axis_label='RSD', y_axis_label="Density", plot_height=height, plot_width=width)
    fig.ygrid.visible = False
    fig.xgrid.visible = False
    fig.title.text_font_size = '16pt'
    fig.xaxis.axis_label_text_font_size = "12pt"
    fig.yaxis.axis_label_text_font_size = "12pt"

    for i in plot:
        for j in batch:
            batch_member_j = np.where(DataTable.Batch.isin([j]), 1, 0)  # only batch
            BatchTable_j = DataTable[batch_member_j == 1]
            x_j = BatchTable_j[peak_list].values
            x_plot = x_j[BatchTable_j.SampleType == i]  # Get X

            # Get color
            if i == 'Blank':
                col = '#00FF00'
            elif i == 'Sample':
                colmap = plt.get_cmap(colormap)
                batch_i = j
                b_rgb = colmap([batch_i])
                b_hex = matplotlib.colors.rgb2hex(b_rgb[0])
                col = b_hex
            elif i == 'QC':
                col = '#FF0000'
            elif i == 'QCW':
                col = '#FFC000'
            elif i == 'QCB':
                col = '#FFFC00'
            elif i == 'QCT':
                col = '#00FFFF'
            else:
                pass

            # Alpha reduce for 'All'
            if i == 'Sample':
                alpha_fig = alpha
            else:
                alpha_fig = alpha / len(batch)

            # For names
            if i == 'Sample':
                i_name = str(j)
            else:
                i_name = i

            # add dist to plot
            add_dist_to_plot(fig, x_plot, sampletype=i_name, col=col, parametric=parametric, padding=padding, smooth=smooth, alpha=alpha_fig)

    # legend
    if legend == False:
        fig.legend.visible = False

    grid.append(fig)  # Add to grid

    # Loop for each batch
    for i in batch:
        fig_i = figure(title="RSD (Batch:{})".format(str(i)), x_axis_label='RSD', y_axis_label="Density", plot_height=height, plot_width=width)
        fig_i.ygrid.visible = False
        fig_i.xgrid.visible = False
        fig_i.title.text_font_size = '16pt'
        fig_i.xaxis.axis_label_text_font_size = "12pt"
        fig_i.yaxis.axis_label_text_font_size = "12pt"

        for j in plot:
            batch_member_j = np.where(DataTable.Batch.isin([i]), 1, 0)  # only batch
            BatchTable_j = DataTable[batch_member_j == 1]
            x_j = BatchTable_j[peak_list].values
            x_plot = x_j[BatchTable_j.SampleType == j]  # Get X

            # Get color
            if j == 'Blank':
                col = '#00FF00'
            elif j == 'Sample':
                colmap = plt.get_cmap(colormap)
                batch_i = i
                b_rgb = colmap([batch_i])
                b_hex = matplotlib.colors.rgb2hex(b_rgb[0])
                col = b_hex
            elif j == 'QC':
                col = '#FF0000'
            elif j == 'QCW':
                col = '#FFC000'
            elif j == 'QCB':
                col = '#FFFC00'
            elif j == 'QCT':
                col = '#00FFFF'
            else:
                pass

            # For names
            if j == 'Sample':
                i_name = str(i)
            else:
                i_name = j

            # add dist to plot
            add_dist_to_plot(fig_i, x_plot, sampletype=i_name, col=col, parametric=parametric, padding=padding, smooth=smooth, alpha=alpha)

        # legend
        if legend == False:
            fig_i.legend.visible = False

        grid.append(fig_i)  # Append fig_i

    output_notebook()
    grid_final = gridplot([grid])
    show(grid_final)


def add_dist_to_plot(fig, x_plot, sampletype, col='#00FFFF', parametric=True, padding=0.05, smooth=None, alpha=0.05):

    # Calculate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if parametric == True:
            RSD = 100 * np.nanstd(x_plot, ddof=1, axis=0) / np.nanmean(x_plot, axis=0)
        else:
            RSD = 100 * 1.4826 * mad(x_plot) / np.nanmedian(x_plot, axis=0)

    RSD = RSD[~np.isnan(RSD)]

    # Get min, max, and padding
    r = np.array(RSD)
    r = np.append(r, [r[0] + 0.05, r[0] - 0.05])
    r_min, r_max = r.min(), r.max()
    r_padding = (r_max - r_min) * padding

    # min r can't be less than 0
    rin_min_padding = r_min - r_padding
    if rin_min_padding < 0:
        rin_min_padding = 0

    # Calculate density
    r_grid = np.linspace(rin_min_padding, r_max + r_padding, 500)
    r_pdf = stats.gaussian_kde(r, smooth)
    r_pdf_grid = r_pdf(r_grid)
    r_grid = np.insert(r_grid, 0, rin_min_padding)
    r_grid = np.insert(r_grid, 0, r_max + r_padding)
    r_pdf_grid = np.insert(r_pdf_grid, 0, 0)
    r_pdf_grid = np.insert(r_pdf_grid, 0, 0)

    # Plot
    fig.patch(r_grid, r_pdf_grid, alpha=alpha, line_alpha=0.8, color=col, line_color=col, line_dash='solid', line_width=2, legend_label=sampletype)


def mad(data):
    return np.nanmedian(np.absolute(data - np.nanmedian(data, axis=0)), axis=0)
