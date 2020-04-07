import numpy as np
import pandas as pd
from random import randint
import matplotlib
import matplotlib.pyplot as plt
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.layouts import gridplot
from bokeh.models.markers import X
from collections import Counter
from bokeh.models.glyphs import Line
from bokeh.models import Label, HoverTool
import warnings
import sys
import os
import sys
from bokeh.io import export_svgs
from .QCRSC import QCRSC
from .control_limits import control_limits
from .table_check import table_check
from .calc_rsd_dratio_blank import calc_rsd_dratio_blank
from .sampletype_binary import sampletype_binary


def peak(DataTable, PeakTable, batch='all', peak='r', gamma='default', transform='log', parametric=True, zero_remove=True, plot='all', control_limit=False, colormap='Accent', fill_points=True, scale_x=1, scale_y=1):

    DataTable = sampletype_binary(DataTable)  # Create binary columns for ['QC', 'QCT', 'Sample', 'Blank']

    if batch == 'all':
        batch = list(DataTable.Batch.unique())  # all batches

    if type(batch) == int:
        batch = [batch]  # put batch in a list

    if peak == 'r':
        peak = PeakTable.Name.sample().values[0]  # random peak

    if peak == 'R':
        peak = PeakTable.Name.sample().values[0]  # random peak

    if gamma == 'default':
        gamma = (0.5, 5, 0.2)  # min 0.5, max 5, step 0.2

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

    # Error check: peak
    if peak not in DataTable.columns.values:
        print("{} is not a peak in the DataTable".format(peak))
        return

    # Error check: gamma
    if gamma == False:
        pass
    elif len(gamma) == 3:
        pass
    else:
        print("gamma must be 'default', False, or list/tuple (3 values) e.g. (0.5, 5, 0.2). This is not correct: {}".format(gamma))
        return

    # Error check: transform
    transform_check = ['log', 'glog', False]
    if transform not in transform_check:
        print("transform must be 'log', 'glog', or False. {} is not an option".format(transform))
        return

    # Error check: parametric
    parametric_check = [True, False]
    if parametric not in parametric_check:
        print("parametric must be True, or False. {} is not an option".format(parametric))
        return

    # Error check: zero remove
    zero_remove_check = [True, False]
    if zero_remove not in zero_remove_check:
        print("zero_remove must be True, or False. {} is not an option".format(zero_remove))
        return

    # Error check: plot
    plot_check = list(DataTable.SampleType.unique())
    for i in plot:
        if i not in plot_check:
            print("plot can only include the following: {}. {} is not an option".format(plot_check, i))
            return

    # Error check: control_limit
    if control_limit == False:
        pass
    elif type(control_limit) == dict:
        control_limit_keys = control_limit.keys()
        for i in control_limit_keys:
            if i not in ['RSD', 'Dratio']:
                print("control_limit '{}' is not an option. Possible options are 'RSD', or 'Dratio".format(i))
                return
        try:
            float(control_limit[i])
        except ValueError:
            print("control_limit dict requires numbers. {} is not a number.".format(control_limit[i]))
            return
    else:
        print("control_limit has to be either False or a dictionary e.g. dict([('RSD', 20)]). {} is not an option".format(control_limit))
        return

    # Error check scale
    if scale_x < 1:
        print("Please use a value greater or equal to 1 for scale_x")
        return

    if scale_y < 1:
        print("Please use a value greater or equal to 1 for scale_y")
        return

    # Error check: fill point
    fill_points_check = [True, False]
    if fill_points not in fill_points_check:
        print("fill_points must be True, or False. {} is not an option".format(zero_remove))
        return

    # Error check: colormap
    colormap_all = "Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r"
    colormap_check = [i.strip() for i in colormap_all.split(',')]
    if colormap not in colormap_check:
        print("{} is not an option for colormap. Possible options include 'Set1', 'Set2', 'rainbow'. All options for colormap can be found here: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html".format(colormap))
        return

    # Plot
    if len(batch) == 1:
        peak_singlebatch(DataTable, PeakTable, batch=batch, peak=peak, gamma=gamma, transform=transform, parametric=parametric, zero_remove=zero_remove, plot=plot, control_limit=control_limit, colormap=colormap, fill_points=fill_points, scale_x=scale_x, scale_y=scale_y)
    else:
        peak_multibatch(DataTable, PeakTable, batch=batch, peak=peak, gamma=gamma, transform=transform, parametric=parametric, zero_remove=zero_remove, plot=plot, control_limit=control_limit, colormap=colormap, scale_x=scale_x, scale_y=scale_y, fill_points=fill_points)


def peak_singlebatch(DataTable, PeakTable, batch='all', peak='M1', gamma=False, transform='log', parametric=True, zero_remove=True, plot=['Sample', 'QC'], control_limit=False, colormap='Set1', fill_points=True, scale_x=1, scale_y=1):

    if gamma != False:
        gamma_range = [x / 100.0 for x in range(int(gamma[0] * 100), int(gamma[1] * 100), int(gamma[2] * 100))]

    index = PeakTable[PeakTable.Name == peak].index[0]  # index of peak
    batch_member = np.where(DataTable.Batch == batch[0], 1, 0)  # only batch
    BatchTable = DataTable[batch_member == 1]

    # Extract and transform data
    x = BatchTable[peak]
    t = BatchTable.Order
    qcw = BatchTable.QCW
    qcb = BatchTable.QCB
    qct = BatchTable.QCT
    sam = BatchTable.Sample
    blank = BatchTable.Blank
    sampletype = BatchTable.SampleType
    batch_list = BatchTable.Batch.values

    # Check for any QCT (True/False)
    qct_check = (qct == 1).any()
    qc_check = (qcw == qcb).all()

    # Remove zeros and tranform
    if zero_remove == True:
        x = x.replace(0, np.nan)
    if transform is 'log':
        x = np.log10(x)

    # Calc RSD, D-ratio, and Blank%Mean
    Before_RSD_within, Before_Dratio_within, Before_Blank_within = calc_rsd_dratio_blank(x, qcw, sam, blank, transform, parametric)
    Before_RSD_between, Before_Dratio_between, Before_Blank_between = calc_rsd_dratio_blank(x, qcb, sam, blank, transform, parametric)
    Before_RSD_test, Before_Dratio_test, Before_Blank_test = calc_rsd_dratio_blank(x, qct, sam, blank, transform, parametric)
    mpa_mean = np.nanmean(x[qcb == True])  # Need to edit

    # perform the QCRSC (if gamma != False)
    if gamma != False:
        z, f, curvetype, cvMse, gamma_optimal, mpa_median = QCRSC(x, t, qcw, gamma_range)
        z = z - mpa_median  # not necessary
        z = z + mpa_mean  # not necessary
        After_RSD_within, After_Dratio_within, After_Blank_within = calc_rsd_dratio_blank(z, qcw, sam, blank, transform, parametric)
        After_RSD_between, After_Dratio_between, After_Blank_between = calc_rsd_dratio_blank(z, qcb, sam, blank, transform, parametric)
        After_RSD_test, After_Dratio_test, After_Blank_test = calc_rsd_dratio_blank(z, qct, sam, blank, transform, parametric)

    # Select what to plot (based on plot)
    plot_binary = BatchTable['SampleType'].isin(plot)
    x = x[plot_binary == True]
    t = t[plot_binary == True]
    sampletype = sampletype[plot_binary == True]
    order = BatchTable.Order[plot_binary == True]
    if gamma != False:
        z = z[plot_binary == True]
        f = np.array(f)
        f = f[plot_binary == True]

    # Create empty grid (2x2)
    grid = np.full((2, 2), None)

    # Set width
    left_width = 300
    right_width = 600
    height = 260

    # If only scale_x is used
    if scale_y == 1:
        if scale_x != 1:
            right_width = int(right_width * scale_x)

    # If only scale_y is used
    if scale_y != 1:
        if scale_x == 1:
            height = int(height * scale_y)

    # If both scale_x and scale_y are ysed
    if scale_y != 1:
        if scale_x != 1:
            height = int(height * scale_y)
            left_width = int(height * scale_y * 0.75)
            right_width = int(right_width * scale_x)

    # Set y_label
    if transform is 'log':
        y_label = 'log(Peak Area)'
    else:
        y_label = 'Peak Area'

    # Get colors
    color_sampletype = BatchTable.SampleType[plot_binary == True].values
    colmap = plt.get_cmap(colormap)
    col = []
    for i in range(len(color_sampletype)):
        if color_sampletype[i] == 'Blank':
            col.append('#00FF00')
        elif color_sampletype[i] == 'Sample':
            batch_i = batch_list[i]
            b_rgb = colmap([batch_i])
            b_hex = matplotlib.colors.rgb2hex(b_rgb[0])
            col.append(b_hex)
        elif color_sampletype[i] == 'QC':
            col.append('#FF0000')
        elif color_sampletype[i] == 'QCW':
            col.append('#FFC000')
        elif color_sampletype[i] == 'QCB':
            col.append('#FFFC00')
        elif color_sampletype[i] == 'QCT':
            col.append('#00FFFF')
        else:
            pass
    col = np.array(col)

    # Before correction plot
    if gamma == False:
        before_title = "{}-{}".format(PeakTable.Name[index], PeakTable.Label[index])
    else:
        before_title = "Before Correction: {}-{}".format(PeakTable.Name[index], PeakTable.Label[index])
    grid[0, 1] = figure(title=before_title, plot_width=right_width, plot_height=height, x_axis_label='Order', y_axis_label=y_label)
    grid[0, 1].ygrid.visible = False
    grid[0, 1].xgrid.visible = False
    grid[0, 1].title.text_font_size = '14pt'

    # Before: Add control limit
    if control_limit != False:
        for i in control_limit:
            low, upp = control_limits(x, qcb, sam, i, control_limit[i], transform)
            before_cl_low = [low] * len(t)
            before_cl_upp = [upp] * len(t)
            if i == 'RSD':
                before_cl_dash = "dashed"
            else:
                before_cl_dash = "dotdash"
            grid[0, 1].line(x=t.values, y=before_cl_low, line_dash=before_cl_dash, line_width=2, line_color='darkblue')
            grid[0, 1].line(x=t.values, y=before_cl_upp, line_dash=before_cl_dash, line_width=2, line_color='darkblue')

    # Before: Plot line ('X' and dash)
    if gamma != False:
        grid[0, 1].x(t.values, f, line_width=2, fill_color=None, line_color='black')
        grid[0, 1].line(t.values, f, line_width=2, line_dash="dashed", line_color='black')
    else:
        x_qc_mean_list = np.ones(len(t)) * mpa_mean
        grid[0, 1].line(t.values, x_qc_mean_list, line_width=2, line_dash="dashed", line_color='black')

    # Before: Plot Samples
    if 'Sample' in sampletype.values:
        plist = (sampletype == 'Sample').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))

        if fill_points == True:
            glyph_before = grid[0, 1].circle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=8, source=source_before)
        else:
            glyph_before = grid[0, 1].circle(x="x", y="y", fill_color="white", fill_alpha=0, line_color="color", size=8, source=source_before)

        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot Blank
    if 'Blank' in sampletype.values:
        plist = (sampletype == 'Blank').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = grid[0, 1].diamond(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QC
    if 'QC' in sampletype.values:
        plist = (sampletype == 'QC').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = grid[0, 1].square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_before)
        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QCW
    if 'QCW' in sampletype.values:
        plist = (sampletype == 'QCW').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = grid[0, 1].square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_before)
        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QCB
    if 'QCB' in sampletype.values:
        plist = (sampletype == 'QCB').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = grid[0, 1].triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QCT
    if 'QCT' in sampletype.values:
        plist = (sampletype == 'QCT').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = grid[0, 1].triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # After and cvMSE plot (if gamma != False)
    if gamma != False:
        # After correction plot
        grid[1, 1] = figure(title="After Correction: {}-{}".format(PeakTable.Name[index], PeakTable.Label[index]), plot_width=right_width, plot_height=height, x_axis_label='Order', y_axis_label=y_label)
        grid[1, 1].ygrid.visible = False
        grid[1, 1].xgrid.visible = False
        grid[1, 1].title.text_font_size = '14pt'

        # After: Add control limit
        if control_limit != False:
            for i in control_limit:
                low, upp = control_limits(z, qcb, sam, i, control_limit[i], transform)
                before_cl_low = [low] * len(t)
                before_cl_upp = [upp] * len(t)
                if i == 'RSD':
                    before_cl_dash = "dashed"
                else:
                    before_cl_dash = "dotdash"
                grid[1, 1].line(x=t.values, y=before_cl_low, line_dash=before_cl_dash, line_width=2, line_color='darkblue')
                grid[1, 1].line(x=t.values, y=before_cl_upp, line_dash=before_cl_dash, line_width=2, line_color='darkblue')

        # After: Plot line (dash)
        x_qc_mean_list = np.ones(len(t)) * mpa_mean
        grid[1, 1].line(t.values, x_qc_mean_list, line_width=2, line_dash="dashed", line_color='black')

        # After: Plot Samples
        if 'Sample' in sampletype.values:
            plist = (sampletype == 'Sample').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))

            if fill_points == True:
                glyph_after = grid[1, 1].circle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=8, source=source_after)
            else:
                glyph_after = grid[1, 1].circle(x="x", y="y", fill_color="white", fill_alpha=0, line_color="color", size=8, source=source_after)

            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # After: Plot Blank
        if 'Blank' in sampletype.values:
            plist = (sampletype == 'Blank').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
            glyph_after = grid[1, 1].diamond(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_after)
            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # After: Plot QC
        if 'QC' in sampletype.values:
            plist = (sampletype == 'QC').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
            glyph_after = grid[1, 1].square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_after)
            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # After: Plot QCW
        if 'QCW' in sampletype.values:
            plist = (sampletype == 'QCW').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
            glyph_after = grid[1, 1].square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_after)
            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # After: Plot QCB
        if 'QCB' in sampletype.values:
            plist = (sampletype == 'QCB').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
            glyph_after = grid[1, 1].triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_after)
            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # After: Plot QCT
        if 'QCT' in sampletype.values:
            plist = (sampletype == 'QCT').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
            glyph_after = grid[1, 1].triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_after)
            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # cvMSE plot
        grid[1, 0] = figure(title="Optimisation Curve", plot_width=left_width, plot_height=height, x_axis_label="γ", y_axis_label="cvMSE")
        grid[1, 0].ygrid.visible = False
        grid[1, 0].xgrid.visible = False
        grid[1, 0].title.text_font_size = '14pt'

        # cvMSE: Get color
        b_rgb = colmap([batch[0]])
        b_hex = matplotlib.colors.rgb2hex(b_rgb[0])

        # cvMSE: Get optimal
        idx_optimal = np.where(np.array(gamma_range) == gamma_optimal)[0]
        cvMSE_optimal = cvMse[idx_optimal]
        grid[1, 0].circle_x(gamma_optimal, cvMSE_optimal, size=12, line_width=1, line_color='#FF00FF', fill_color='white', alpha=1)

        # cvMSE: Plot circles and lines
        source_cvMSE = ColumnDataSource(dict(x=gamma_range, y=cvMse))
        grid[1, 0].line(x="x", y="y", line_color="grey", line_width=1, source=source_cvMSE)
        glyph_cvMSE_circle = grid[1, 0].circle(x="x", y="y", fill_color=b_hex, line_color="grey", fill_alpha=1, size=5, source=source_cvMSE)

        # cvMSE: Add HoverTool
        grid[1, 0].add_tools(HoverTool(
            renderers=[glyph_cvMSE_circle],
            tooltips=[
                ("γ", "@x"),
                ("cvMSE", "@y"), ],))

    # Text box
    grid[0, 0] = figure(title="", plot_width=left_width, plot_height=(height + 5), x_axis_label="", y_axis_label="", outline_line_alpha=0)
    grid[0, 0].circle(0, 0, line_color='white', fill_color='white', fill_alpha=0)  # Necessary to remove warning
    grid[0, 0].xaxis.visible = False
    grid[0, 0].yaxis.visible = False
    grid[0, 0].ygrid.visible = False
    grid[0, 0].xgrid.visible = False

    # Text box: labels
    text_label = []
    text_label.append('Batch: {}'.format(batch[0]))
    text_label.append('Name: {}'.format(PeakTable.Name[index]))
    text_label.append('Label: {}'.format(PeakTable.Label[index]))
    if transform is 'log':
        text_label.append('log(MPA): {0:.2f}'.format(mpa_mean))
    else:
        text_label.append('MPA: {0:.2f}'.format(mpa_mean))
    if gamma == False:
        if qc_check == True:
            if qct_check == False:
                text_label.append('RSD: {0:.2f}'.format(Before_RSD_within))
                text_label.append('D-Ratio: {0:.2f}'.format(Before_Dratio_within))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
            else:
                text_label.append('RSD Train: {0:.2f}'.format(Before_RSD_within))
                text_label.append('RSD Test: {0:.2f}'.format(Before_RSD_test))
                text_label.append('D-Ratio Train: {0:.2f}'.format(Before_Dratio_within))
                text_label.append('D-Ratio Test: {0:.2f}'.format(Before_Dratio_test))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
        else:
            if qct_check == False:
                text_label.append('RSD Within: {0:.2f}'.format(Before_RSD_within))
                text_label.append('RSD Between: {0:.2f}'.format(Before_RSD_between))
                text_label.append('D-Ratio Within: {0:.2f}'.format(Before_Dratio_within))
                text_label.append('D-Ratio Between: {0:.2f}'.format(Before_Dratio_between))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
            else:
                text_label.append('RSD Within: {0:.2f}'.format(Before_RSD_within))
                text_label.append('RSD Between: {0:.2f}'.format(Before_RSD_between))
                text_label.append('RSD Test: {0:.2f}'.format(Before_RSD_test))
                text_label.append('D-Ratio Within: {0:.2f}'.format(Before_Dratio_within))
                text_label.append('D-Ratio Between: {0:.2f}'.format(Before_Dratio_between))
                text_label.append('D-Ratio Test: {0:.2f}'.format(Before_Dratio_test))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
    else:
        if qc_check == True:
            if qct_check == False:
                text_label.append('Correction method: {}'.format(curvetype))
                text_label.append('Optimal γ: {}'.format(gamma_optimal))
                text_label.append('RSD [BEFORE]AFTER: [{}] {}'.format(np.round(Before_RSD_within, 2), np.round(After_RSD_within, 2)))
                text_label.append('D-Ratio [BEFORE]AFTER: [{}] {}'.format(np.round(Before_Dratio_within, 2), np.round(After_Dratio_within, 2)))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
            else:
                text_label.append('Correction method: {}'.format(curvetype))
                text_label.append('Optimal γ: {}'.format(gamma_optimal))
                text_label.append('RSD Train [BEFORE]AFTER: [{}] {}'.format(np.round(Before_RSD_within, 2), np.round(After_RSD_within, 2)))
                text_label.append('RSD Test [BEFORE]AFTER: [{}] {}'.format(np.round(Before_RSD_test, 2), np.round(After_RSD_test, 2)))
                text_label.append('D-Ratio Train [BEFORE]AFTER: [{}] {}'.format(np.round(Before_Dratio_within, 2), np.round(After_Dratio_within, 2)))
                text_label.append('D-Ratio Test [BEFORE]AFTER: [{}] {}'.format(np.round(Before_Dratio_test, 2), np.round(After_Dratio_test, 2)))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
        else:
            if qct_check == False:
                text_label.append('Correction method: {}'.format(curvetype))
                text_label.append('Optimal γ: {}'.format(gamma_optimal))
                text_label.append('RSD Within [BEF]AFT: [{}] {}'.format(np.round(Before_RSD_within, 2), np.round(After_RSD_within, 2)))
                text_label.append('RSD Between [BEF]AFT: [{}] {}'.format(np.round(Before_RSD_between, 2), np.round(After_RSD_between, 2)))
                text_label.append('D-Ratio Within [BEF]AFT: [{}] {}'.format(np.round(Before_Dratio_within, 2), np.round(After_Dratio_within, 2)))
                text_label.append('D-Ratio Between [BEF]AFT: [{}] {}'.format(np.round(Before_Dratio_between, 2), np.round(After_Dratio_between, 2)))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
            else:
                text_label.append('Correction method: {}'.format(curvetype))
                text_label.append('Optimal γ: {}'.format(gamma_optimal))
                text_label.append('RSD Within [BEF]AFT: [{}] {}'.format(np.round(Before_RSD_within, 2), np.round(After_RSD_within, 2)))
                text_label.append('RSD Between [BEF]AFT: [{}] {}'.format(np.round(Before_RSD_between, 2), np.round(After_RSD_between, 2)))
                text_label.append('RSD Test [BEF]AFT: [{}] {}'.format(np.round(Before_RSD_test, 2), np.round(After_RSD_test, 2)))
                text_label.append('D-Ratio Within [BEF]AFT: [{}] {}'.format(np.round(Before_Dratio_within, 2), np.round(After_Dratio_within, 2)))
                text_label.append('D-Ratio Between [BEF]AFT: [{}] {}'.format(np.round(Before_Dratio_between, 2), np.round(After_Dratio_between, 2)))
                text_label.append('D-Ratio Test [BEF]AFT: [{}] {}'.format(np.round(Before_Dratio_test, 2), np.round(After_Dratio_test, 2)))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))

    if gamma == False:
        if qc_check == True:
            if qct_check == False:
                text_x = [70] * 7
                text_y = [210, 182, 154, 126, 98, 70, 42]
                text_size = ['9.1pt'] * 7
            else:
                text_x = [70] * 9
                text_y = [210, 189, 168, 147, 126, 105, 84, 63, 42]
                text_size = ['9pt'] * 9
        else:
            if qct_check == False:
                text_x = [70] * 9
                text_y = [210, 189, 168, 147, 126, 105, 84, 63, 42]
                text_size = ['9pt'] * 9
            else:
                text_x = [70] * 11
                text_y = [210, 193, 176, 159, 142, 125, 108, 91, 74, 57, 40]
                text_size = ['8pt'] * 11
    else:
        if qc_check == True:
            if qct_check == False:
                text_x = [0] * 9
                text_y = [210, 189, 168, 147, 126, 105, 84, 63, 42]
                text_size = ['9pt'] * 9
            else:
                text_x = [0] * 11
                text_y = [210, 193, 176, 159, 142, 125, 108, 91, 74, 57, 40]
                text_size = ['8pt'] * 11
        else:
            if qct_check == False:
                text_x = [0] * 11
                text_y = [210, 193, 176, 159, 142, 125, 108, 91, 74, 57, 40]
                text_size = ['8pt'] * 11
            else:
                text_x = [0] * 13
                text_y = [210, 196, 182, 168, 154, 140, 126, 112, 98, 84, 70, 56, 42]
                text_size = ['8pt'] * 13

    # Add all text
    for i in range(len(text_label)):
        label = Label(x=text_x[i], y=text_y[i], x_units='screen', y_units='screen', text=text_label[i], text_font_size=text_size[i])
        grid[0, 0].add_layout(label)

    # Show figure
    output_notebook()
    fig = gridplot(grid.tolist())
    show(fig)


def peak_multibatch(DataTable, PeakTable, batch='all', peak='M1', gamma=False, transform='log', parametric=True, zero_remove=True, plot=['Sample', 'QC'], control_limit=False, colormap='Set1', fill_points=True, scale_x=1, scale_y=1):

    if gamma != False:
        gamma_range = [x / 100.0 for x in range(int(gamma[0] * 100), int(gamma[1] * 100), int(gamma[2] * 100))]

    index = PeakTable[PeakTable.Name == peak].index[0]  # index of peak
    batch_member = np.where(DataTable.Batch.isin(batch), 1, 0)  # only batch
    BatchTable = DataTable[batch_member == 1]

    # Extract and transform data
    x = BatchTable[peak]
    t = BatchTable.Order
    qcw = BatchTable.QCW
    qcb = BatchTable.QCB
    qct = BatchTable.QCT
    sam = BatchTable.Sample
    blank = BatchTable.Blank
    sampletype = BatchTable.SampleType
    batch_list = BatchTable.Batch.values

    # Check for any QCT (True/False)
    qct_check = (qct == 1).any()
    qc_check = (qcw == qcb).all()

    # Remove zeros and tranform
    if zero_remove == True:
        x = x.replace(0, np.nan)
    if transform is 'log':
        x = np.log10(x)

    # Calc RSD, D-ratio, and Blank%Mean
    Before_RSD_within, Before_Dratio_within, Before_Blank_within = calc_rsd_dratio_blank(x, qcw, sam, blank, transform, parametric)
    Before_RSD_between, Before_Dratio_between, Before_Blank_between = calc_rsd_dratio_blank(x, qcb, sam, blank, transform, parametric)
    Before_RSD_test, Before_Dratio_test, Before_Blank_test = calc_rsd_dratio_blank(x, qct, sam, blank, transform, parametric)
    mpa_mean = np.nanmedian(x[qcb == True])  # Need to edit

    # perform the QCRSC (if gamma != False)
    if gamma != False:
        z = []
        f = []
        curvetype = []
        cvMse = []
        gamma_optimal = []
        mpa_median = []
        for i in batch:
            batch_i = np.where(batch_list == i, True, False)
            x_i = x[batch_i]
            t_i = t[batch_i]
            qcw_i = qcw[batch_i]
            qcb_i = qcb[batch_i]
            z_i, f_i, curvetype_i, cvMse_i, gamma_optimal_i, mpa_median_i = QCRSC(x_i, t_i, qcw_i, gamma_range)
            mpa_median_i = np.nanmedian(z_i.values[qcb_i == 1])
            z_i = z_i - mpa_median_i  # align z
            z.append(z_i)
            f.append(f_i)
            curvetype.append(curvetype_i)
            cvMse.append(cvMse_i)
            gamma_optimal.append(gamma_optimal_i)
            mpa_median.append(mpa_median_i)
        z = np.array(np.concatenate(z, axis=0))
        z = z + np.nanmedian(mpa_median)
        f = np.array(np.concatenate(f, axis=0))

        mpa_mean = np.nanmedian(mpa_median)  # Need to edit

        After_RSD_within, After_Dratio_within, After_Blank_within = calc_rsd_dratio_blank(z, qcw, sam, blank, transform, parametric)
        After_RSD_between, After_Dratio_between, After_Blank_between = calc_rsd_dratio_blank(z, qcb, sam, blank, transform, parametric)
        After_RSD_test, After_Dratio_test, After_Blank_test = calc_rsd_dratio_blank(z, qct, sam, blank, transform, parametric)

    # Select what to plot (based on plot)
    plot_binary = BatchTable['SampleType'].isin(plot)
    x = x[plot_binary == True]
    t = t[plot_binary == True]
    sampletype = sampletype[plot_binary == True]
    order = BatchTable.Order[plot_binary == True]
    if gamma != False:
        z = z[plot_binary == True]
        f = np.array(f)
        f = f[plot_binary == True]

    # Create empty grid (2x2)
    grid = np.full((2, 2), None)

    # Set width
    left_width = 300
    right_width = 600
    height = 260

    # If only scale_x is used
    if scale_y == 1:
        if scale_x != 1:
            right_width = int(right_width * scale_x)

    # If only scale_y is used
    if scale_y != 1:
        if scale_x == 1:
            height = int(height * scale_y)

    # If both scale_x and scale_y are ysed
    if scale_y != 1:
        if scale_x != 1:
            height = int(height * scale_y)
            left_width = int(height * scale_y * 0.75)
            right_width = int(right_width * scale_x)

    # Set y_label
    if transform is 'log':
        y_label = 'log(Peak Area)'
    else:
        y_label = 'Peak Area'

    # Get colors
    color_sampletype = BatchTable.SampleType[plot_binary == True].values
    colmap = plt.get_cmap(colormap)
    col = []
    for i in range(len(color_sampletype)):
        if color_sampletype[i] == 'Blank':
            col.append('#00FF00')
        elif color_sampletype[i] == 'Sample':
            batch_i = batch_list[i]
            b_rgb = colmap([batch_i])
            b_hex = matplotlib.colors.rgb2hex(b_rgb[0])
            col.append(b_hex)
        elif color_sampletype[i] == 'QC':
            col.append('#FF0000')
        elif color_sampletype[i] == 'QCW':
            col.append('#FFC000')
        elif color_sampletype[i] == 'QCB':
            col.append('#FFFC00')
        elif color_sampletype[i] == 'QCT':
            col.append('#00FFFF')
        else:
            pass
    col = np.array(col)

    # Before correction plot
    if gamma == False:
        before_title = "{}-{}".format(PeakTable.Name[index], PeakTable.Label[index])
    else:
        before_title = "Before Correction: {}-{}".format(PeakTable.Name[index], PeakTable.Label[index])
    grid[0, 1] = figure(title=before_title, plot_width=right_width, plot_height=height, x_axis_label='Order', y_axis_label=y_label)
    grid[0, 1].ygrid.visible = False
    grid[0, 1].xgrid.visible = False
    grid[0, 1].title.text_font_size = '14pt'

    # Before: Add control limit
    if control_limit != False:
        for i in control_limit:
            for j in batch:
                batch_j = np.where(batch_list == j, True, False)
                x_j = x.values[batch_j]
                qcb_j = qcb.values[batch_j]
                sam_j = sam.values[batch_j]
                t_j = t.values[batch_j]
                low, upp = control_limits(x_j, qcb_j, sam_j, i, control_limit[i], transform)
                before_cl_low = [low] * len(t_j)
                before_cl_upp = [upp] * len(t_j)
                if i == 'RSD':
                    before_cl_dash = "dashed"
                else:
                    before_cl_dash = "dotdash"
                grid[0, 1].line(x=t_j, y=before_cl_low, line_dash=before_cl_dash, line_width=2, line_color='darkblue')
                grid[0, 1].line(x=t_j, y=before_cl_upp, line_dash=before_cl_dash, line_width=2, line_color='darkblue')

    # Before: Plot line ('X' and dash)
    if gamma != False:
        grid[0, 1].x(t.values, f, line_width=2, fill_color=None, line_color='black')
        grid[0, 1].line(t.values, f, line_width=2, line_dash="dashed", line_color='black')
    else:
        x_qc_mean_list = np.ones(len(t)) * mpa_mean
        grid[0, 1].line(t.values, x_qc_mean_list, line_width=2, line_dash="dashed", line_color='black')

    # Before: Plot Samples
    if 'Sample' in sampletype.values:
        plist = (sampletype == 'Sample').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))

        if fill_points == True:
            glyph_before = grid[0, 1].circle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=8, source=source_before)
        else:
            glyph_before = grid[0, 1].circle(x="x", y="y", fill_color="white", fill_alpha=0, line_color="color", size=8, source=source_before)

        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot Blank
    if 'Blank' in sampletype.values:
        plist = (sampletype == 'Blank').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = grid[0, 1].diamond(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QC
    if 'QC' in sampletype.values:
        plist = (sampletype == 'QC').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = grid[0, 1].square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_before)
        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QCW
    if 'QCW' in sampletype.values:
        plist = (sampletype == 'QCW').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = grid[0, 1].square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_before)
        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QCB
    if 'QCB' in sampletype.values:
        plist = (sampletype == 'QCB').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = grid[0, 1].triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QCT
    if 'QCT' in sampletype.values:
        plist = (sampletype == 'QCT').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x.values[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = grid[0, 1].triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
        grid[0, 1].add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # After and cvMSE plot (if gamma != False)
    if gamma != False:
        # After correction plot
        grid[1, 1] = figure(title="After Correction: {}-{}".format(PeakTable.Name[index], PeakTable.Label[index]), plot_width=right_width, plot_height=height, x_axis_label='Order', y_axis_label=y_label)
        grid[1, 1].ygrid.visible = False
        grid[1, 1].xgrid.visible = False
        grid[1, 1].title.text_font_size = '14pt'

        # After: Add control limit
        if control_limit != False:
            for i in control_limit:
                low, upp = control_limits(z, qcb, sam, i, control_limit[i], transform)
                before_cl_low = [low] * len(t)
                before_cl_upp = [upp] * len(t)
                if i == 'RSD':
                    before_cl_dash = "dashed"
                else:
                    before_cl_dash = "dotdash"
                grid[1, 1].line(x=t.values, y=before_cl_low, line_dash=before_cl_dash, line_width=2, line_color='darkblue')
                grid[1, 1].line(x=t.values, y=before_cl_upp, line_dash=before_cl_dash, line_width=2, line_color='darkblue')

        # After: Plot line (dash)
        x_qc_mean_list = np.ones(len(t)) * mpa_mean
        grid[1, 1].line(t.values, x_qc_mean_list, line_width=2, line_dash="dashed", line_color='black')

        # After: Plot Samples
        if 'Sample' in sampletype.values:
            plist = (sampletype == 'Sample').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))

            if fill_points == True:
                glyph_after = grid[1, 1].circle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=8, source=source_after)
            else:
                glyph_after = grid[1, 1].circle(x="x", y="y", fill_color="white", fill_alpha=0, line_color="color", size=8, source=source_after)

            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # After: Plot Blank
        if 'Blank' in sampletype.values:
            plist = (sampletype == 'Blank').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
            glyph_after = grid[1, 1].diamond(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_after)
            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # After: Plot QC
        if 'QC' in sampletype.values:
            plist = (sampletype == 'QC').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
            glyph_after = grid[1, 1].square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_after)
            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # After: Plot QCW
        if 'QCW' in sampletype.values:
            plist = (sampletype == 'QCW').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
            glyph_after = grid[1, 1].square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_after)
            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # After: Plot QCB
        if 'QCB' in sampletype.values:
            plist = (sampletype == 'QCB').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
            glyph_after = grid[1, 1].triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_after)
            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # After: Plot QCT
        if 'QCT' in sampletype.values:
            plist = (sampletype == 'QCT').values
            source_after = ColumnDataSource(dict(x=t.values[plist], y=z[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
            glyph_after = grid[1, 1].triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_after)
            grid[1, 1].add_tools(HoverTool(
                renderers=[glyph_after],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # cvMSE plot
        grid[1, 0] = figure(title="Optimisation Curve", plot_width=left_width, plot_height=height, x_axis_label="γ", y_axis_label="cvMSE")
        grid[1, 0].ygrid.visible = False
        grid[1, 0].xgrid.visible = False
        grid[1, 0].title.text_font_size = '14pt'

        # cvMSE: Per Batch
        for i in range(len(cvMse)):
            # cvMSE: Get color
            b_rgb = colmap([batch[i]])
            b_hex = matplotlib.colors.rgb2hex(b_rgb[0])

            # cvMSE: Get optimal
            idx_optimal = np.where(np.array(gamma_range) == gamma_optimal[i])[0]
            cvMSE_optimal = cvMse[i][idx_optimal]
            grid[1, 0].circle_x(gamma_optimal[i], cvMSE_optimal, size=12, line_width=1, line_color='#FF00FF', fill_color='white', alpha=1)

            # cvMSE: Plot circles and lines
            source_cvMSE = ColumnDataSource(dict(x=gamma_range, y=cvMse[i]))
            grid[1, 0].line(x="x", y="y", line_color="grey", line_width=1, source=source_cvMSE)
            glyph_cvMSE_circle = grid[1, 0].circle(x="x", y="y", fill_color=b_hex, line_color="grey", fill_alpha=0.9, size=5, source=source_cvMSE)

            # cvMSE: Add HoverTool
            grid[1, 0].add_tools(HoverTool(
                renderers=[glyph_cvMSE_circle],
                tooltips=[
                    ("γ", "@x"),
                    ("cvMSE", "@y"), ],))

    # Text box
    grid[0, 0] = figure(title="", plot_width=left_width, plot_height=(height + 5), x_axis_label="", y_axis_label="", outline_line_alpha=0)
    grid[0, 0].circle(0, 0, line_color='white', fill_color='white', fill_alpha=0)  # Necessary to remove warning
    grid[0, 0].xaxis.visible = False
    grid[0, 0].yaxis.visible = False
    grid[0, 0].ygrid.visible = False
    grid[0, 0].xgrid.visible = False

    # Text box: labels
    text_label = []
    text_label.append('Batch: {}'.format(batch[0]))
    text_label.append('Name: {}'.format(PeakTable.Name[index]))
    text_label.append('Label: {}'.format(PeakTable.Label[index]))
    if transform is 'log':
        text_label.append('log(MPA): {0:.2f}'.format(mpa_mean))
    else:
        text_label.append('MPA: {0:.2f}'.format(mpa_mean))
    if gamma == False:
        if qc_check == True:
            if qct_check == False:
                text_label.append('RSD: {0:.2f}'.format(Before_RSD_within))
                text_label.append('D-Ratio: {0:.2f}'.format(Before_Dratio_within))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
            else:
                text_label.append('RSD Train: {0:.2f}'.format(Before_RSD_within))
                text_label.append('RSD Test: {0:.2f}'.format(Before_RSD_test))
                text_label.append('D-Ratio Train: {0:.2f}'.format(Before_Dratio_within))
                text_label.append('D-Ratio Test: {0:.2f}'.format(Before_Dratio_test))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
        else:
            if qct_check == False:
                text_label.append('RSD Within: {0:.2f}'.format(Before_RSD_within))
                text_label.append('RSD Between: {0:.2f}'.format(Before_RSD_between))
                text_label.append('D-Ratio Within: {0:.2f}'.format(Before_Dratio_within))
                text_label.append('D-Ratio Between: {0:.2f}'.format(Before_Dratio_between))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
            else:
                text_label.append('RSD Within: {0:.2f}'.format(Before_RSD_within))
                text_label.append('RSD Between: {0:.2f}'.format(Before_RSD_between))
                text_label.append('RSD Test: {0:.2f}'.format(Before_RSD_test))
                text_label.append('D-Ratio Within: {0:.2f}'.format(Before_Dratio_within))
                text_label.append('D-Ratio Between: {0:.2f}'.format(Before_Dratio_between))
                text_label.append('D-Ratio Test: {0:.2f}'.format(Before_Dratio_test))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
    else:
        if qc_check == True:
            if qct_check == False:
                text_label.append('Correction method: {}'.format(curvetype))
                text_label.append('Optimal γ: {}'.format(gamma_optimal))
                text_label.append('RSD [BEFORE]AFTER: [{}] {}'.format(np.round(Before_RSD_within, 2), np.round(After_RSD_within, 2)))
                text_label.append('D-Ratio [BEFORE]AFTER: [{}] {}'.format(np.round(Before_Dratio_within, 2), np.round(After_Dratio_within, 2)))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
            else:
                text_label.append('Correction method: {}'.format(curvetype))
                text_label.append('Optimal γ: {}'.format(gamma_optimal))
                text_label.append('RSD Train [BEFORE]AFTER: [{}] {}'.format(np.round(Before_RSD_within, 2), np.round(After_RSD_within, 2)))
                text_label.append('RSD Test [BEFORE]AFTER: [{}] {}'.format(np.round(Before_RSD_test, 2), np.round(After_RSD_test, 2)))
                text_label.append('D-Ratio Train [BEFORE]AFTER: [{}] {}'.format(np.round(Before_Dratio_within, 2), np.round(After_Dratio_within, 2)))
                text_label.append('D-Ratio Test [BEFORE]AFTER: [{}] {}'.format(np.round(Before_Dratio_test, 2), np.round(After_Dratio_test, 2)))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
        else:
            if qct_check == False:
                text_label.append('Correction method: {}'.format(curvetype))
                text_label.append('Optimal γ: {}'.format(gamma_optimal))
                text_label.append('RSD Within [BEF]AFT: [{}] {}'.format(np.round(Before_RSD_within, 2), np.round(After_RSD_within, 2)))
                text_label.append('RSD Between [BEF]AFT: [{}] {}'.format(np.round(Before_RSD_between, 2), np.round(After_RSD_between, 2)))
                text_label.append('D-Ratio Within [BEF]AFT: [{}] {}'.format(np.round(Before_Dratio_within, 2), np.round(After_Dratio_within, 2)))
                text_label.append('D-Ratio Between [BEF]AFT: [{}] {}'.format(np.round(Before_Dratio_between, 2), np.round(After_Dratio_between, 2)))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))
            else:
                text_label.append('Correction method: {}'.format(curvetype))
                text_label.append('Optimal γ: {}'.format(gamma_optimal))
                text_label.append('RSD Within [BEF]AFT: [{}] {}'.format(np.round(Before_RSD_within, 2), np.round(After_RSD_within, 2)))
                text_label.append('RSD Between [BEF]AFT: [{}] {}'.format(np.round(Before_RSD_between, 2), np.round(After_RSD_between, 2)))
                text_label.append('RSD Test [BEF]AFT: [{}] {}'.format(np.round(Before_RSD_test, 2), np.round(After_RSD_test, 2)))
                text_label.append('D-Ratio Within [BEF]AFT: [{}] {}'.format(np.round(Before_Dratio_within, 2), np.round(After_Dratio_within, 2)))
                text_label.append('D-Ratio Between [BEF]AFT: [{}] {}'.format(np.round(Before_Dratio_between, 2), np.round(After_Dratio_between, 2)))
                text_label.append('D-Ratio Test [BEF]AFT: [{}] {}'.format(np.round(Before_Dratio_test, 2), np.round(After_Dratio_test, 2)))
                text_label.append('Blank%Mean: {0:.2f}'.format(Before_Blank_within))

    if gamma == False:
        if qc_check == True:
            if qct_check == False:
                text_x = [70] * 7
                text_y = [210, 182, 154, 126, 98, 70, 42]
                text_size = ['9.1pt'] * 7
            else:
                text_x = [70] * 9
                text_y = [210, 189, 168, 147, 126, 105, 84, 63, 42]
                text_size = ['9pt'] * 9
        else:
            if qct_check == False:
                text_x = [70] * 9
                text_y = [210, 189, 168, 147, 126, 105, 84, 63, 42]
                text_size = ['9pt'] * 9
            else:
                text_x = [70] * 11
                text_y = [210, 193, 176, 159, 142, 125, 108, 91, 74, 57, 40]
                text_size = ['8pt'] * 11
    else:
        if qc_check == True:
            if qct_check == False:
                text_x = [0] * 9
                text_y = [210, 189, 168, 147, 126, 105, 84, 63, 42]
                text_size = ['9pt'] * 9
            else:
                text_x = [0] * 11
                text_y = [210, 193, 176, 159, 142, 125, 108, 91, 74, 57, 40]
                text_size = ['8pt'] * 11
        else:
            if qct_check == False:
                text_x = [0] * 11
                text_y = [210, 193, 176, 159, 142, 125, 108, 91, 74, 57, 40]
                text_size = ['8pt'] * 11
            else:
                text_x = [0] * 13
                text_y = [210, 196, 182, 168, 154, 140, 126, 112, 98, 84, 70, 56, 42]
                text_size = ['8pt'] * 13

    # Add all text
    for i in range(len(text_label)):
        label = Label(x=text_x[i], y=text_y[i], x_units='screen', y_units='screen', text=text_label[i], text_font_size=text_size[i])
        grid[0, 0].add_layout(label)

    # Show figure
    output_notebook()
    fig = gridplot(grid.tolist())
    show(fig)
