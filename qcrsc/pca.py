import numpy as np
import pandas as pd
from sklearn import decomposition
from random import randint
import matplotlib
from copy import deepcopy
import matplotlib.pyplot as plt
from bokeh.models.markers import X
from bokeh.plotting import ColumnDataSource, figure
from bokeh.models import Label, HoverTool
from collections import Counter
from bokeh.plotting import output_notebook, show, figure
from bokeh.layouts import gridplot, column, row
from bokeh.models.glyphs import Line
from bokeh.models import Label, HoverTool
from .sampletype_binary import sampletype_binary
from .control_limits import control_limits
from .ci95_ellipse import ci95_ellipse
from .QCRSC import QCRSC
from .table_check import table_check
from .calc_rsd_dratio import calc_rsd_dratio
from .calc_rsd_dratio_blank import calc_rsd_dratio_blank
from .scale import scale_values
from .knnimpute import knnimpute


def pca_plot(DataTable, PeakTable, pcx=1, pcy=2, batch='all', project_qc=True, transform='log', zero_remove=True, plot='all', control_limit=False, scale='unit', knn=3, colormap='Accent', color_batches=True, plot_ellipse='all', plot_points=True, scale_x=1, scale_y=1, fill_points=True, alpha_ellipse=(0.1, 0.1)):

    DataTable = sampletype_binary(DataTable)  # Create binary columns for ['QC', 'QCT', 'Sample', 'Blank']

    if type(pcx) != int:
        print("pcx must be an integer.")

    if type(pcy) != int:
        print("pcy must be an integer.")

    if type(knn) != int:
        print("knn must be an integer.")

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

    # Error check: project_qc
    project_qc_check = [True, False]
    if project_qc not in project_qc_check:
        print("parametric must be True, or False. {} is not an option".format(parametric))
        return

    # Error check: transform
    transform_check = ['log', 'glog', False]
    if transform not in transform_check:
        print("transform must be 'log', 'glog', or False. {} is not an option".format(transform))
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

    # Error check: scale
    scale_check = ['auto', 'unit', 'pareto', 'vast', 'level', False]
    if scale not in scale_check:
        print("scale must be 'auto', 'unit', 'pareto', 'vast', 'level' or False. {} is not an option".format(zero_remove))
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

    # Error check: plot points
    plot_points_check = [True, False]
    if plot_points not in plot_points_check:
        print("plot_points must be True, or False. {} is not an option".format(zero_remove))
        return

    # Error check: plot ellipse
    plot_ellipse_check = ['all', 'none', 'meanci', 'popci']
    if plot_ellipse not in plot_ellipse_check:
        print("plot_ellipse must be 'all', 'none', 'meanci', or 'popci'. {} is not an option".format(zero_remove))
        return

    # Error check: color batches
    color_batches_check = [True, False]
    if color_batches not in color_batches_check:
        print("color_batches must be True or False. {} is not an option".format(zero_remove))
        return

    # Error check: colormap
    colormap_all = "Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r"
    colormap_check = [i.strip() for i in colormap_all.split(',')]
    if colormap not in colormap_check:
        print("{} is not an option for colormap. Possible options include 'Set1', 'Set2', 'rainbow'. All options for colormap can be found here: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html".format(colormap))
        return

    # Plot
    if len(batch) == 1:
        pca_singlebatch(DataTable, PeakTable, pcx=pcx, pcy=pcy, batch=batch, project_qc=project_qc, transform=transform, zero_remove=zero_remove, plot=plot, control_limit=control_limit, scale=scale, knn=knn, colormap=colormap, color_batches=color_batches, plot_ellipse=plot_ellipse, plot_points=plot_points, scale_x=scale_x, scale_y=scale_y, fill_points=fill_points, alpha_ellipse=alpha_ellipse)


def pca_singlebatch(DataTable, PeakTable, pcx=1, pcy=2, batch='all', project_qc=True, transform='log', zero_remove=True, plot='all', control_limit=False, scale='unit', knn=3, colormap='Set3', color_batches=True, plot_ellipse='all', plot_points=True, scale_x=1, scale_y=1, fill_points=True, alpha_ellipse=(0.1, 0.1)):

    peak_list = PeakTable.Name
    batch_member = np.where(DataTable.Batch == batch[0], 1, 0)  # only batch
    BatchTable = DataTable[batch_member == 1]

    # Set width
    scatter_width = 350
    scatter_height = 360
    peak_width = 500
    peak_height = 170

    # If only scale_x is used
    if scale_y == 1:
        if scale_x != 1:
            peak_width = int(peak_width * scale_x)

    # If only scale_y is used
    if scale_y != 1:
        if scale_x == 1:
            scatter_height = int(scatter_height * scale_y)
            peak_height = int(peak_height * scale_y)

    # If both scale_x and scale_y are ysed
    if scale_y != 1:
        if scale_x != 1:
            scatter_width = int(scatter_width * scale_y)
            peak_width = int(peak_width * scale_x)
            scatter_height = int(scatter_height * scale_y)
            peak_height = int(peak_height * scale_y)

    # Extract and transform data
    x = BatchTable[peak_list]
    t = BatchTable.Order
    qcw = BatchTable.QCW
    qcb = BatchTable.QCB
    qct = BatchTable.QCT
    sam = BatchTable.Sample
    blank = BatchTable.Blank
    sampletype = BatchTable.SampleType
    batch_list = BatchTable.Batch.values

    # Remove zeros and tranform
    if zero_remove == True:
        x = x.replace(0, np.nan)
    if transform is 'log':
        x = np.log10(x)
    if scale != False:
        x = scale_values(x.values, method=scale)
    else:
        x = np.array(x)
    x = knnimpute(x, k=knn)

    # Set model
    model = decomposition.PCA()
    if project_qc == True:
        model.fit(x[sam == 1])
    else:
        model.fit(x)
    scores_ = model.transform(x)
    explained_var_ = model.explained_variance_ratio_ * 100

    # Extract scores, explained variance, and loadings for pcx and pcy
    x_score = scores_[:, (pcx - 1)]
    y_score = scores_[:, (pcy - 1)]
    x_expvar = explained_var_[(pcx - 1)]
    y_expvar = explained_var_[(pcy - 1)]
    x_load = model.components_[(pcx - 1), :]
    y_load = model.components_[(pcy - 1), :]

    # To plot
    plot_binary = BatchTable['SampleType'].isin(plot)
    x_score_plot = x_score[plot_binary == True]
    y_score_plot = y_score[plot_binary == True]
    sampletype = sampletype[plot_binary == True]
    t = t[plot_binary == True]

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

    # Scores plot
    fig_score = figure(title="", x_axis_label="PC {} ({:0.1f}%)".format(pcx, x_expvar), y_axis_label="PC {} ({:0.1f}%)".format(pcy, y_expvar), plot_width=scatter_width, plot_height=scatter_height)
    fig_score.ygrid.visible = False
    fig_score.xgrid.visible = False
    fig_score.title.text_font_size = '14pt'

    # plot points
    if plot_points == True:
        # Plot Samples
        if 'Sample' in sampletype.values:
            plist = (sampletype == 'Sample').values
            source_before = ColumnDataSource(dict(x=x_score_plot[plist], y=y_score_plot[plist], label=sampletype[plist], color=col[plist], Name=t[plist]))

            if fill_points == True:
                glyph_before = fig_score.circle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=8, source=source_before)
            else:
                glyph_before = fig_score.circle(x="x", y="y", fill_color="white", fill_alpha=0, line_color="color", size=8, source=source_before)

            fig_score.add_tools(HoverTool(
                renderers=[glyph_before],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # Before: Plot Blank
        if 'Blank' in sampletype.values:
            plist = (sampletype == 'Blank').values
            source_before = ColumnDataSource(dict(x=x_score_plot[plist], y=y_score_plot[plist], label=sampletype[plist], color=col[plist], Name=t[plist]))
            glyph_before = fig_score.diamond(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
            fig_score.add_tools(HoverTool(
                renderers=[glyph_before],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # Before: Plot QC
        if 'QC' in sampletype.values:
            plist = (sampletype == 'QC').values
            source_before = ColumnDataSource(dict(x=x_score_plot[plist], y=y_score_plot[plist], label=sampletype[plist], color=col[plist], Name=t[plist]))
            glyph_before = fig_score.square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_before)
            fig_score.add_tools(HoverTool(
                renderers=[glyph_before],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # Before: Plot QCW
        if 'QCW' in sampletype.values:
            plist = (sampletype == 'QCW').values
            source_before = ColumnDataSource(dict(x=x_score_plot[plist], y=y_score_plot[plist], label=sampletype[plist], color=col[plist], Name=t[plist]))
            glyph_before = fig_score.square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_before)
            fig_score.add_tools(HoverTool(
                renderers=[glyph_before],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # Before: Plot QCB
        if 'QCB' in sampletype.values:
            plist = (sampletype == 'QCB').values
            source_before = ColumnDataSource(dict(x=x_score_plot[plist], y=y_score_plot[plist], label=sampletype[plist], color=col[plist], Name=t[plist]))
            glyph_before = fig_score.triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
            fig_score.add_tools(HoverTool(
                renderers=[glyph_before],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

        # Before: Plot QCT
        if 'QCT' in sampletype.values:
            plist = (sampletype == 'QCT').values
            source_before = ColumnDataSource(dict(x=x_score_plot[plist], y=y_score_plot[plist], label=sampletype[plist], color=col[plist], Name=t[plist]))
            glyph_before = fig_score.triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
            fig_score.add_tools(HoverTool(
                renderers=[glyph_before],
                tooltips=[
                    ("Type", "@label"),
                    ("Order", "@Name"), ],))

    unique_sampletype = np.unique(sampletype)
    list_color = []
    for i in unique_sampletype:
        if i == 'Blank':
            list_color.append('#00FF00')
        elif i == 'Sample':
            batch_i = batch[0]
            b_rgb = colmap([batch_i])
            b_hex = matplotlib.colors.rgb2hex(b_rgb[0])
            list_color.append(b_hex)
        elif i == 'QC':
            list_color.append('#FF0000')
        elif i == 'QCW':
            list_color.append('#FFC000')
        elif i == 'QCB':
            list_color.append('#FFFC00')
        elif i == 'QCT':
            list_color.append('#00FFFF')

    # Add 95% confidence ellipse for each unique group in a loop
    for i in range(len(unique_sampletype)):
        # Get scores for the corresponding group
        group_i_x = []
        group_i_y = []
        for j in range(len(sampletype)):
            if sampletype.values[j] == unique_sampletype[i]:
                group_i_x.append(x_score_plot[j])
                group_i_y.append(y_score_plot[j])

        if unique_sampletype[i] == 'Sample':
            alpha_ellipse_plot = alpha_ellipse[0]
        else:
            alpha_ellipse_plot = alpha_ellipse[1]

        # Calculate ci95 ellipse for each group
        data_circ_group = pd.DataFrame({"0": group_i_x, "1": group_i_y})
        m, outside_m = ci95_ellipse(data_circ_group, type="mean")
        p, outside_p = ci95_ellipse(data_circ_group, type="pop")

        # Plot ci95 ellipse shade and line
        if plot_ellipse == 'all':
            fig_score.line(m[:, 0], m[:, 1], color=list_color[i], line_width=1, alpha=0.8, line_dash="solid")
            fig_score.line(p[:, 0], p[:, 1], color=list_color[i], alpha=0.4)
            fig_score.patch(m[:, 0], m[:, 1], color=list_color[i], alpha=alpha_ellipse_plot)
            fig_score.patch(p[:, 0], p[:, 1], color=list_color[i], alpha=alpha_ellipse_plot / 2)
            fig_score.x(np.median(m[:, 0]), np.median(m[:, 1]), color=list_color[i], size=5, alpha=0.6, line_width=1.5)
        elif plot_ellipse == 'meanci':
            fig_score.line(m[:, 0], m[:, 1], color=list_color[i], line_width=1, alpha=0.8, line_dash="solid")
            fig_score.patch(m[:, 0], m[:, 1], color=list_color[i], alpha=alpha_ellipse_plot)
            fig_score.x(np.median(m[:, 0]), np.median(m[:, 1]), color=list_color[i], size=5, alpha=0.6, line_width=1.5)
        elif plot_ellipse == 'popci':
            fig_score.line(p[:, 0], p[:, 1], color=list_color[i], alpha=0.4)
            fig_score.patch(p[:, 0], p[:, 1], color=list_color[i], alpha=alpha_ellipse_plot / 2)
            fig_score.x(np.median(m[:, 0]), np.median(m[:, 1]), color=list_color[i], size=5, alpha=0.6, line_width=1.5)
        else:
            pass

    # Clean later ... -> boxplot'
    group_sampletype = sampletype.values
    group_label = group_sampletype
    cats = np.unique(group_label)

    # generate some synthetic time series for six different categories
    df = pd.DataFrame(dict(score=y_score_plot, group=group_label))

    # find the quartiles and IQR for each category
    groups = df.groupby('group')
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    # find the outliers for each category
    def outliers(group):
        cat = group.name
        return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
    out = groups.apply(outliers).dropna()

    # prepare outlier data for plotting, we need coordinates for every outlier.
    if not out.empty:
        outx = []
        outy = []
        for keys in out.index:
            outx.append(keys[0])
            outy.append(out.loc[keys[0]].loc[keys[1]])

    p = figure(x_range=cats, y_range=fig_score.y_range, toolbar_location=None, width=70, height=scatter_height)

    # # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.score = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, 'score']), upper.score)]
    lower.score = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, 'score']), lower.score)]

    # stems
    p.segment(cats, upper.score, cats, q3.score, line_color="black")
    p.segment(cats, lower.score, cats, q1.score, line_color="black")

    # boxes
    p.vbar(cats, 0.4, q2.score, q3.score, fill_color=list_color, line_color="black", alpha=0.3)
    p.vbar(cats, 0.4, q1.score, q2.score, fill_color=list_color, line_color="black", alpha=0.3)

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(cats, lower.score, 0.2, 0.01, line_color="black")
    p.rect(cats, upper.score, 0.2, 0.01, line_color="black")

    # outliers
    if not out.empty:
        p.circle(outx, outy, size=3, color="orange", fill_alpha=0.6)

    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False

    # generate some synthetic time series for six different categories
    df = pd.DataFrame(dict(score=x_score_plot, group=group_label))

    # find the quartiles and IQR for each category
    groups = df.groupby('group')
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    # find the outliers for each category
    def outliers(group):
        cat = group.name
        return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
    out = groups.apply(outliers).dropna()

    # prepare outlier data for plotting, we need coordinates for every outlier.
    if not out.empty:
        outx = []
        outy = []
        for keys in out.index:
            outx.append(keys[0])
            outy.append(out.loc[keys[0]].loc[keys[1]])

    g = figure(y_range=cats, x_range=fig_score.x_range, toolbar_location=None, width=scatter_width, height=70)

    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.score = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, 'score']), upper.score)]
    lower.score = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, 'score']), lower.score)]

    # stems
    g.segment(upper.score, cats, q3.score, cats, line_color="black")
    g.segment(lower.score, cats, q1.score, cats, line_color="black")

    # boxes
    g.hbar(cats, 0.4, q2.score, q3.score, fill_color=list_color, line_color="black", alpha=0.3)
    g.hbar(cats, 0.4, q1.score, q2.score, fill_color=list_color, line_color="black", alpha=0.3)

    # whiskers (almost-0 height rects simpler than segments)
    g.rect(lower.score, cats, 0.01, 0.2, line_color="black")
    g.rect(upper.score, cats, 0.01, 0.2, line_color="black")

    # outliers
    if not out.empty:
        g.circle(outy, outx, size=3, color="orange", fill_alpha=0.6)

    g.xgrid.visible = False
    g.ygrid.visible = False
    g.xaxis.visible = False
    g.yaxis.visible = False

    fig_score.min_border_left = 0
    fig_score.min_border_right = 0
    fig_score.min_border_top = 0
    fig_score.min_border_bottom = 0

    g.min_border_top = 20

    # # Now create a control chart
    pc1 = pca_controlchart_singlebatch(DataTable, PeakTable, x_score, batch, plot=plot, control_limit=control_limit, colormap=colormap, width=peak_width, height=peak_height, fill_point=fill_points, y_label="PC {} ({:0.1f}%)".format(pcx, x_expvar))

    pc2 = pca_controlchart_singlebatch(DataTable, PeakTable, y_score, batch, plot=plot, control_limit=control_limit, colormap=colormap, width=peak_width, height=peak_height, fill_point=fill_points, y_label="PC {} ({:0.1f}%)".format(pcy, y_expvar))

    # Add Text
    text_fig = figure(title="", plot_width=70, plot_height=50, x_axis_label="", y_axis_label="", outline_line_alpha=0)
    text_fig2 = figure(title="", plot_width=500, plot_height=50, x_axis_label="", y_axis_label="", outline_line_alpha=0)

    text1 = Label(x=40, y=--2, x_units='screen', y_units='screen', text='No. of Peaks: {}'.format((len(PeakTable))), text_font_size='16pt')
    text2 = Label(x=300, y=--2, x_units='screen', y_units='screen', text='MV D-ratio: xx.xx', text_font_size='16pt')
    text_fig2.add_layout(text1)
    text_fig2.add_layout(text2)
    text_fig.circle(0, 0, line_color='white', fill_color='white', fill_alpha=0)  # Necessary to remove warning
    text_fig2.circle(0, 0, line_color='white', fill_color='white', fill_alpha=0)
    text_fig.xaxis.visible = False
    text_fig.yaxis.visible = False
    text_fig.ygrid.visible = False
    text_fig.xgrid.visible = False
    text_fig2.xaxis.visible = False
    text_fig2.yaxis.visible = False
    text_fig2.ygrid.visible = False
    text_fig2.xgrid.visible = False

    # Plot all figs
    pc_control = column(children=[pc1, pc2], sizing_mode='scale_height')
    fig = gridplot([[pc1], [pc2]])
    fig = gridplot([[g, text_fig, text_fig2], [fig_score, p, pc_control]])
    output_notebook()
    show(fig)


def pca_controlchart_singlebatch(DataTable, PeakTable, x_score, batch, plot='all', control_limit=False, colormap='plasma', y_label='y_label', width=10, height=10, fill_point=True):

    batch_member = np.where(DataTable.Batch == batch[0], 1, 0)  # only batch
    BatchTable = DataTable[batch_member == 1]

    # Extract and transform data
    x = x_score
    x_save = deepcopy(x)
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

    # MPA mean
    mpa_mean = np.nanmean(x[qcb == True])  # Need to edit

    # Select what to plot (based on plot)
    plot_binary = BatchTable['SampleType'].isin(plot)
    x = x[plot_binary == True]
    t = t[plot_binary == True]
    sampletype = sampletype[plot_binary == True]
    order = BatchTable.Order[plot_binary == True]

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
    fig = figure(title="", plot_width=width, plot_height=height, x_axis_label='Order', y_axis_label=y_label)
    fig.ygrid.visible = False
    fig.xgrid.visible = False
    fig.title.text_font_size = '14pt'

    # Before: Add control limit
    if control_limit != False:
        for i in control_limit:
            low, upp = control_limits(x_save, qcb, sam, i, control_limit[i], transform=False)
            before_cl_low = [low] * len(t)
            before_cl_upp = [upp] * len(t)
            if i == 'RSD':
                before_cl_dash = "dashed"
            else:
                before_cl_dash = "dotdash"
            fig.line(x=t.values, y=before_cl_low, line_dash=before_cl_dash, line_width=2, line_color='darkblue')
            fig.line(x=t.values, y=before_cl_upp, line_dash=before_cl_dash, line_width=2, line_color='darkblue')

    # Before: Plot line ('X' and dash)
    x_qc_mean_list = np.ones(len(t)) * mpa_mean
    fig.line(t.values, x_qc_mean_list, line_width=2, line_dash="dashed", line_color='black')

    # Before: Plot Samples
    if 'Sample' in sampletype.values:
        plist = (sampletype == 'Sample').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))

        if fill_point == True:
            glyph_before = fig.circle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=8, source=source_before)
        else:
            glyph_before = fig.circle(x="x", y="y", fill_color="white", fill_alpha=0, line_color="color", size=8, source=source_before)

        fig.add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot Blank
    if 'Blank' in sampletype.values:
        plist = (sampletype == 'Blank').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = fig.diamond(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
        fig.add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QC
    if 'QC' in sampletype.values:
        plist = (sampletype == 'QC').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = fig.square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_before)
        fig.add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QCW
    if 'QCW' in sampletype.values:
        plist = (sampletype == 'QCW').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = fig.square(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=9, source=source_before)
        fig.add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QCB
    if 'QCB' in sampletype.values:
        plist = (sampletype == 'QCB').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = fig.triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
        fig.add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    # Before: Plot QCT
    if 'QCT' in sampletype.values:
        plist = (sampletype == 'QCT').values
        source_before = ColumnDataSource(dict(x=t.values[plist], y=x[plist], label=sampletype[plist], color=col[plist], Name=order[plist]))
        glyph_before = fig.triangle(x="x", y="y", fill_color="color", fill_alpha=1, line_color='grey', size=12, source=source_before)
        fig.add_tools(HoverTool(
            renderers=[glyph_before],
            tooltips=[
                ("Type", "@label"),
                ("Order", "@Name"), ],))

    return fig
