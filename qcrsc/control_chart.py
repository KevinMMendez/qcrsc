import numpy as np
import pandas as pd
from random import randint
import matplotlib
import matplotlib.pyplot as plt
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.layouts import gridplot
from bokeh.models.markers import X
from bokeh.models.glyphs import Line
from bokeh.models import Label, HoverTool
from .QCRSC import QCRSC
from .table_check import table_check
from .calc_rsd_dratio import calc_rsd_dratio


def control_chart(DataTable, PeakTable, batch, peak, gamma='default', transform='log', parametric=True, zeroflag=True, plot=['Sample', 'QC'], control_limit=False, colormap='Set2'):

    if type(batch) == int:
        control_chart_singlebatch(DataTable, PeakTable, batch, peak, gamma=gamma, transform=transform, parametric=parametric, zeroflag=zeroflag, plot=plot, control_limit=control_limit)
    elif len(batch) == 1:
        control_chart_singlebatch(DataTable, PeakTable, batch[0], peak, gamma=gamma, transform=transform, parametric=parametric, zeroflag=zeroflag, plot=plot, control_limit=control_limit)
    else:
        control_chart_multibatch(DataTable, PeakTable, batch, peak, gamma=gamma, transform=transform, parametric=parametric, zeroflag=zeroflag, plot=plot, control_limit=control_limit, colormap=colormap)


def control_chart_singlebatch(DataTable, PeakTable, batch, peak, gamma='default', transform='log', parametric=True, zeroflag=True, plot=['Sample', 'QC'], control_limit=False):

    table_check(DataTable, print_statement=False)
    peak_list = PeakTable.Name

    # Create a QC column based on SampleType (if it doesn't exist)
    qc_col = pd.get_dummies(DataTable.SampleType).QC
    try:
        DataTable.insert(3, 'QC', qc_col)
    except ValueError:
        pass

    # Create a Sample column based on SampleType (if it doesn't exist)
    sam_col = pd.get_dummies(DataTable.SampleType).Sample
    try:
        DataTable.insert(3, 'Sample', sam_col)
    except ValueError:
        pass

    # Create a Blank column based on SampleType (if it doesn't exist)
    try:
        blank_col = pd.get_dummies(DataTable.SampleType).Blank
    except AttributeError:
        blank_col = [0] * len(DataTable)  # No blanks
    try:
        DataTable.insert(4, 'Blank', blank_col)
    except ValueError:
        pass

    # Default gamma_range (Temporary add False)
    if gamma in ['default', False]:
        gamma_input = (0.5, 5, 0.2)
    else:
        gamma_input = gamma

    gamma_range = [x / 100.0 for x in range(int(gamma_input[0] * 100), int(gamma_input[1] * 100), int(gamma_input[2] * 100))]

    # Randomly select a peak if "peak = R"
    if peak is 'R':
        pp = randint(0, len(peak_list) - 1)  # Needs to be -1
        peak = peak_list[pp]

    if len(peak_list[peak_list == peak]) == 0:
        raise ValueError("Fatal Error: peak {} does not exist.".format(peak))

    index = peak_list[peak_list == peak].index[0]

    if batch == -1:
        batch_member = np.ones(len(DataTable.Batch))  # Consider it 1 batch
    else:
        bb = np.unique(DataTable.Batch)
        if batch not in bb:
            raise ValueError("Fatal Error: batch {} does not exist".format(batch))
        batch_member = np.where(DataTable.Batch == batch, 1, 0)

    # Extract and transform data
    x = DataTable[peak]
    x = x[batch_member == 1]
    t = DataTable.Order[batch_member == 1]
    qc = DataTable.QC[batch_member == 1]
    sam = DataTable.Sample[batch_member == 1]
    BatchTable = DataTable[batch_member == 1]
    sampletype = DataTable.SampleType[batch_member == 1]

    if zeroflag == True:
        x = x.replace(0, np.nan)
    if transform is 'log':
        x = np.log10(x)

    # perform the QCRSC
    z, f, curvetype, cvMse, gamma_final, mpa = QCRSC(x, t, qc, gamma_range)

    # Calc RSD and D-ratio
    Before_RSD_QC, Before_RSD_Sam, Before_Dratio = calc_rsd_dratio(x, qc, sam, transform, parametric)
    After_RSD_QC, After_RSD_Sam, After_Dratio = calc_rsd_dratio(z, qc, sam, transform, parametric)

    # Calc Blank peak area ratio (Before & After)
    blank_bpar = DataTable[DataTable.Blank == 1][peak]
    if len(blank_bpar) != 0:
        if transform is 'log':
            before_qc_bpar = np.power(10, x[qc == 1])
            after_qc_bpar = np.power(10, z[qc == 1])
        else:
            before_qc_bpar = x[qc == 1]
            after_qc_bpar = z[qc == 1]

        if parametric == True:
            BPAR_Blank = np.nanmean(blank_bpar)
            Before_BPAR_QC = np.nanmean(before_qc_bpar)
            After_BPAR_QC = np.nanmean(after_qc_bpar)
        else:
            BPAR_Blank = np.nanmedian(blank_bpar)
            Before_BPAR_QC = np.nanmedian(before_qc_bpar)
            After_BPAR_QC = np.nanmedian(after_qc_bpar)

        Before_BPAR = BPAR_Blank / Before_BPAR_QC * 100
        After_BPAR = BPAR_Blank / After_BPAR_QC * 100
    else:
        Before_BPAR = np.nan
        After_BPAR = np.nan

    # Information for control limit (before correction)
    x_qc_mean = np.nanmean(x[qc == 1])
    x_qc_std = np.nanstd(x[qc == 1], ddof=1)
    x_qc_rsd = x_qc_std / x_qc_mean * 100
    x_sam = x[sam == 0]

    # Control limit boundaries (before correction)
    if control_limit == False:
        pass
    elif control_limit[0] == 'D-ratio':
        std_sam = np.nanstd(x_sam, ddof=1)
        std_qc = control_limit[1] * std_sam / 100
        before_control_limit_low = x_qc_mean - 2 * std_qc
        before_control_limit_upp = x_qc_mean + 2 * std_qc
    elif control_limit[0] == 'RSD':
        std_for_rsd = control_limit[1] * np.nanmean(x[qc == 1]) / 100 * (x_qc_rsd / Before_RSD_QC)  # Temporary (deal with log)
        before_control_limit_low = x_qc_mean - 2 * std_for_rsd
        before_control_limit_upp = x_qc_mean + 2 * std_for_rsd
    else:
        raise ValueError("Control limit must be either False, ('RSD', value), or ('D-ratio', value)")

    # Information for control limit (after correction)
    z_qc_mean = np.nanmean(z[qc == 1])
    z_qc_std = np.nanstd(z[qc == 1], ddof=1)
    z_qc_rsd = z_qc_std / z_qc_mean * 100
    z_sam = z[sam == 0]

    # Control limit boundaries (after correction)
    if control_limit == False:
        pass
    elif control_limit[0] == 'D-ratio':
        std_sam = np.nanstd(z_sam, ddof=1)
        std_qc = control_limit[1] * std_sam / 100
        after_control_limit_low = z_qc_mean - 2 * std_qc
        after_control_limit_upp = z_qc_mean + 2 * std_qc
    elif control_limit[0] == 'RSD':
        std_for_rsd = control_limit[1] * np.nanmean(z[qc == 1]) / 100 * (z_qc_rsd / After_RSD_QC)  # Temporary (deal with log)
        after_control_limit_low = z_qc_mean - 2 * std_for_rsd
        after_control_limit_upp = z_qc_mean + 2 * std_for_rsd
    else:
        raise ValueError("Control limit must be either False, ('RSD', value), or ('D-ratio', value)")

    ##################################################################################
    #### Plot using BOKEH ####

    output_notebook()

    # Select what to plot
    plot_binary = BatchTable['SampleType'].isin(plot)
    x = x[plot_binary == True]
    t = t[plot_binary == True]
    z = z[plot_binary == True]
    f = np.array(f)
    f = f[plot_binary == True]
    sampletype = sampletype[plot_binary == True]
    order = BatchTable.Order[plot_binary == True]

    # Create empty grid (2x2)
    grid = np.full((2, 2), None)

    # Set y_label
    if transform is 'log':
        y_label = 'log(Peak Area)'
    else:
        y_label = 'Peak Area'

    # Get colors
    color_sampletype = BatchTable.SampleType[plot_binary == True].values
    col = []
    for i in color_sampletype:
        if i == 'Blank':
            col.append('#00FF00')
        elif i == 'Sample':
            col.append('#00FFFF')
        elif i == 'QC':
            col.append('#FF0000')
        else:
            pass

    # Before correction plot
    grid[0, 1] = figure(title="Batch {} {}:{}".format(batch, PeakTable.Name[index], PeakTable.Label[index]), plot_width=600, plot_height=260, x_axis_label='Order', y_axis_label=y_label)
    grid[0, 1].title.text_font_size = '14pt'

    # Before: Plot line ('X' and dash)
    if gamma != False:
        source_before_line = ColumnDataSource(dict(x=t.values, y=f))
        glyph_before_x = X(x="x", y="y", line_width=2, fill_color=None)
        glyph_before_line = Line(x="x", y="y", line_width=2, line_dash="dashed")
        grid[0, 1].add_glyph(source_before_line, glyph_before_x)
        grid[0, 1].add_glyph(source_before_line, glyph_before_line)
    else:
        source_before_line = ColumnDataSource(dict(x=t.values, y=np.ones(len(t)) * x_qc_mean))
        glyph_before_line = Line(x="x", y="y", line_width=2, line_dash="dashed")
        grid[0, 1].add_glyph(source_before_line, glyph_before_line)

    # # Before: Plot circles
    source_before_circle = ColumnDataSource(dict(x=t.values, y=x.values, label=sampletype, color=col, Name=order))
    glyph_before_circle = grid[0, 1].circle(x="x", y="y", fill_color="color", fill_alpha=1, size=8, source=source_before_circle)

    # # Before: Add HoverTool
    grid[0, 1].add_tools(HoverTool(
        renderers=[glyph_before_circle],
        tooltips=[
            ("Type", "@label"),
            ("Order", "@Name"), ],))

    # # Before: Add control limit
    if control_limit != False:
        if control_limit[0] in ['D-ratio', 'RSD']:
            if np.isnan(before_control_limit_low):  # Can't draw line if it doesn't exist
                pass
            else:
                before_control_limit_low = [before_control_limit_low] * len(t)
                before_control_limit_upp = [before_control_limit_upp] * len(t)
                source_before_control_limit = ColumnDataSource(dict(x=t.values, low=before_control_limit_low, upp=before_control_limit_upp))
                glyph_low = Line(x="x", y="low", line_width=2, line_dash="dashed", line_color='black')
                glyph_upp = Line(x="x", y="upp", line_width=2, line_dash="dashed", line_color='black')
                grid[0, 1].add_glyph(source_before_control_limit, glyph_low)
                grid[0, 1].add_glyph(source_before_control_limit, glyph_upp)

    if gamma != False:
        # cvMSE plot
        grid[1, 0] = figure(title="", plot_width=300, plot_height=260, x_axis_label="γ", y_axis_label="cvMSE")

        # cvMSE: Plot line
        source_cvMSE = ColumnDataSource(dict(x=gamma_range, y=cvMse))
        glyph_cvMSE_line = Line(x="x", y="y", line_color="#0000FF", line_width=2)
        grid[1, 0].add_glyph(source_cvMSE, glyph_cvMSE_line)

        # cvMSE: Plot circles
        glyph_cvMSE_circle = grid[1, 0].circle(x="x", y="y", fill_color="#FF0000", line_color="#0000FF", fill_alpha=1, size=5, source=source_cvMSE)

        # cvMSE: Add HoverTool
        grid[1, 0].add_tools(HoverTool(
            renderers=[glyph_cvMSE_circle],
            tooltips=[
                ("γ", "@x"),
                ("cvMSE", "@y"), ],))

        grid[1, 0].xgrid.visible = False
        grid[1, 0].ygrid.visible = False

        text_x = 0
    else:
        curvetype = 'nan'
        gamma_final = 'nan'
        text_x = 72

    # Textbox
    grid[0, 0] = figure(title="", plot_width=300, plot_height=265, x_axis_label="", y_axis_label="", outline_line_alpha=0)

    text1 = Label(x=text_x, y=210, x_units='screen', y_units='screen', text='Batch: {}'.format(batch), text_font_size='7.5pt')
    text2 = Label(x=text_x, y=190, x_units='screen', y_units='screen', text='Name: {}'.format(PeakTable.Name[index]), text_font_size='7.5pt')
    text3 = Label(x=text_x, y=170, x_units='screen', y_units='screen', text='Label: {}'.format(PeakTable.Label[index]), text_font_size='7.5pt')
    if transform is 'log':
        text4 = Label(x=text_x, y=150, x_units='screen', y_units='screen', text='log(MPA): {}'.format(np.round(mpa, 2)), text_font_size='7.5pt')
    else:
        text4 = Label(x=text_x, y=150, x_units='screen', y_units='screen', text='MPA: {}'.format(np.round(mpa, 2)), text_font_size='7.5pt')
    text5 = Label(x=text_x, y=130, x_units='screen', y_units='screen', text='Correction method: {}'.format(curvetype), text_font_size='7.5pt')
    text6 = Label(x=text_x, y=110, x_units='screen', y_units='screen', text='Optimal γ: {}'.format(gamma_final), text_font_size='7.5pt')
    text7 = Label(x=text_x, y=90, x_units='screen', y_units='screen', text='QC %RSD: {}'.format(np.round(Before_RSD_QC, 2)),
                  text_font_size='7.5pt')
    text8 = Label(x=text_x, y=70, x_units='screen', y_units='screen', text='Sam %RSD: {}'.format(np.round(Before_RSD_Sam, 2)),
                  text_font_size='7.5pt')
    text9 = Label(x=text_x, y=50, x_units='screen', y_units='screen', text='D-Ratio: {}'.format(np.round(Before_Dratio, 2), ),
                  text_font_size='7.5pt')
    text10 = Label(x=text_x, y=30, x_units='screen', y_units='screen', text='Blank-Ratio: {}'.format(np.round(Before_BPAR, 2), ),
                   text_font_size='7.5pt')

    grid[0, 0].add_layout(text1)
    grid[0, 0].add_layout(text2)
    grid[0, 0].add_layout(text3)
    grid[0, 0].add_layout(text4)
    grid[0, 0].add_layout(text5)
    grid[0, 0].add_layout(text6)
    grid[0, 0].add_layout(text7)
    grid[0, 0].add_layout(text8)
    grid[0, 0].add_layout(text9)
    grid[0, 0].add_layout(text10)
    grid[0, 0].circle(0, 0, line_color='white', fill_color='white', fill_alpha=0)  # Necessary to remove warning
    grid[0, 0].xaxis.visible = False
    grid[0, 0].yaxis.visible = False
    grid[0, 0].ygrid.visible = False
    grid[0, 0].xgrid.visible = False

    grid[0, 1].xgrid.visible = False
    grid[0, 1].ygrid.visible = False

    # Show figure
    fig = gridplot(grid.tolist())
    show(fig)


def control_chart_multibatch(DataTable, PeakTable, batch, peak, gamma='default', transform='log', parametric=True, zeroflag=True, plot=['Sample', 'QC'], control_limit=False, colormap='Set2'):

    table_check(DataTable, print_statement=False)
    peak_list = PeakTable.Name

    # Create a QC column based on SampleType (if it doesn't exist)
    qc_col = pd.get_dummies(DataTable.SampleType).QC
    try:
        DataTable.insert(3, 'QC', qc_col)
    except ValueError:
        pass

    # Create a Sample column based on SampleType (if it doesn't exist)
    sam_col = pd.get_dummies(DataTable.SampleType).Sample
    try:
        DataTable.insert(3, 'Sample', sam_col)
    except ValueError:
        pass

    # Create a Blank column based on SampleType (if it doesn't exist)
    try:
        blank_col = pd.get_dummies(DataTable.SampleType).Blank
    except AttributeError:
        blank_col = [0] * len(DataTable)  # No blanks
    try:
        DataTable.insert(4, 'Blank', blank_col)
    except ValueError:
        pass

    # Default gamma_range (Temporary add False)
    if gamma in ['default', False]:
        gamma_input = (0.5, 5, 0.2)
    else:
        gamma_input = gamma

    gamma_range = [x / 100.0 for x in range(int(gamma_input[0] * 100), int(gamma_input[1] * 100), int(gamma_input[2] * 100))]

    # Randomly select a peak if "peak = R"
    if peak is 'R':
        pp = randint(0, len(peak_list) - 1)  # Needs to be -1
        peak = peak_list[pp]

    if len(peak_list[peak_list == peak]) == 0:
        raise ValueError("Fatal Error: peak {} does not exist.".format(peak))

    index = peak_list[peak_list == peak].index[0]

    batch_i = batch[0]
    bb = np.unique(DataTable.Batch)
    if batch_i not in bb:
        raise ValueError("Fatal Error: batch {} does not exist".format(batch_i))
    batch_member = np.where(DataTable.Batch == batch_i, 1, 0)

    # Extract and transform data
    x = DataTable[peak]
    x = x[batch_member == 1]
    t = DataTable.Order[batch_member == 1]
    qc = DataTable.QC[batch_member == 1]
    b = DataTable.Batch[batch_member == 1]
    sam = DataTable.Sample[batch_member == 1]
    BatchTable = DataTable[batch_member == 1]
    sampletype = DataTable.SampleType[batch_member == 1]

    if zeroflag == True:
        x = x.replace(0, np.nan)
    if transform is 'log':
        x = np.log10(x)

    # perform the QCRSC
    z, f, curvetype, cvMse, gamma_final, mpa = QCRSC(x, t, qc, gamma_range)
    gamma_final = [gamma_final]
    curvetype = [curvetype]
    mpa = [mpa]
    z_list = []
    z_list.append(z.values)

    for i in batch[1:]:
        batch_i = i
        bb = np.unique(DataTable.Batch)
        if batch_i not in bb:
            raise ValueError("Fatal Error: batch {} does not exist".format(batch_i))
        batch_member = np.where(DataTable.Batch == batch_i, 1, 0)

        # Extract and transform data
        x_i = DataTable[peak]
        x_i = x_i[batch_member == 1]
        t_i = DataTable.Order[batch_member == 1]
        qc_i = DataTable.QC[batch_member == 1]
        b_i = DataTable.Batch[batch_member == 1]
        sam_i = DataTable.Sample[batch_member == 1]
        BatchTable_i = DataTable[batch_member == 1]
        sampletype_i = DataTable.SampleType[batch_member == 1]

        if zeroflag == True:
            x_i = x_i.replace(0, np.nan)
        if transform is 'log':
            x_i = np.log10(x_i)

        #   perform the QCRSC
        z_i, f_i, curvetype_i, cvMse_i, gamma_final_i, mpa_i = QCRSC(x_i, t_i, qc_i, gamma_range)
        x = x.append(x_i)
        t = t.append(t_i)
        qc = qc.append(qc_i)
        sam = sam.append(sam_i)
        b = b.append(b_i)
        BatchTable = BatchTable.append(BatchTable_i)
        sampletype = sampletype.append(sampletype_i)

        f = np.concatenate([f, f_i])
        z_list.append(z_i.values)
        gamma_final = np.concatenate([gamma_final, [gamma_final_i]])
        curvetype = np.concatenate([curvetype, [curvetype_i]])
        mpa = np.concatenate([mpa, [mpa_i]])

    b = b.values

    # Align
    z = []
    for i in range(len(batch)):
        z_align = z_list[i]
        mpa_align = mpa[i]

        # if transform is 'log':
        #     z_align = np.log10(z_align)
        #     mpa_align = np.log10(mpa_align)

        z_align = z_align - mpa_align
        for j in z_align:
            z.append(j)
    z = pd.Series(z, index=x.index)
    mpa = np.nanmedian(mpa, axis=0)
    z = z + np.array(mpa)

    # Calc RSD and D-ratio
    Before_RSD_QC, Before_RSD_Sam, Before_Dratio = calc_rsd_dratio(x, qc, sam, transform, parametric)
    After_RSD_QC, After_RSD_Sam, After_Dratio = calc_rsd_dratio(z, qc, sam, transform, parametric)

    # Calc Blank peak area ratio (Before & After)
    blank_bpar = DataTable[DataTable.Blank == 1][peak]
    if len(blank_bpar) != 0:
        if transform is 'log':
            before_qc_bpar = np.power(10, x[qc == 1])
            after_qc_bpar = np.power(10, z[qc == 1])
        else:
            before_qc_bpar = x[qc == 1]
            after_qc_bpar = z[qc == 1]

        if parametric == True:
            BPAR_Blank = np.nanmean(blank_bpar)
            Before_BPAR_QC = np.nanmean(before_qc_bpar)
            After_BPAR_QC = np.nanmean(after_qc_bpar)
        else:
            BPAR_Blank = np.nanmedian(blank_bpar)
            Before_BPAR_QC = np.nanmedian(before_qc_bpar)
            After_BPAR_QC = np.nanmedian(after_qc_bpar)

        Before_BPAR = BPAR_Blank / Before_BPAR_QC * 100
        After_BPAR = BPAR_Blank / After_BPAR_QC * 100
    else:
        Before_BPAR = np.nan
        After_BPAR = np.nan

    # Information for control limit (before correction)
    x_qc_mean = np.nanmean(x[qc == 1])
    x_qc_std = np.nanstd(x[qc == 1], ddof=1)
    x_qc_rsd = x_qc_std / x_qc_mean * 100
    x_sam = x[sam == 0]

    # Control limit boundaries (before correction)
    if control_limit == False:
        pass
    elif control_limit[0] == 'D-ratio':
        std_sam = np.nanstd(x_sam, ddof=1)
        std_qc = control_limit[1] * std_sam / 100
        before_control_limit_low = x_qc_mean - 2 * std_qc
        before_control_limit_upp = x_qc_mean + 2 * std_qc
    elif control_limit[0] == 'RSD':
        std_for_rsd = control_limit[1] * np.nanmean(x[qc == 1]) / 100 * (x_qc_rsd / Before_RSD_QC)  # Temporary (deal with log)
        before_control_limit_low = x_qc_mean - 2 * std_for_rsd
        before_control_limit_upp = x_qc_mean + 2 * std_for_rsd
    else:
        raise ValueError("Control limit must be either False, ('RSD', value), or ('D-ratio', value)")

    # Information for control limit (after correction)
    z_qc_mean = np.nanmean(z[qc == 1])
    z_qc_std = np.nanstd(z[qc == 1], ddof=1)
    z_qc_rsd = z_qc_std / z_qc_mean * 100
    z_sam = z[sam == 0]

    # Control limit boundaries (after correction)
    if control_limit == False:
        pass
    elif control_limit[0] == 'D-ratio':
        std_sam = np.nanstd(z_sam, ddof=1)
        std_qc = control_limit[1] * std_sam / 100
        after_control_limit_low = z_qc_mean - 2 * std_qc
        after_control_limit_upp = z_qc_mean + 2 * std_qc
    elif control_limit[0] == 'RSD':
        std_for_rsd = control_limit[1] * np.nanmean(z[qc == 1]) / 100 * (z_qc_rsd / After_RSD_QC)  # Temporary (deal with log)
        after_control_limit_low = z_qc_mean - 2 * std_for_rsd
        after_control_limit_upp = z_qc_mean + 2 * std_for_rsd
    else:
        raise ValueError("Control limit must be either False, ('RSD', value), or ('D-ratio', value)")

    ##################################################################################
    #### Plot using BOKEH ####

    output_notebook()

    # Select what to plot
    plot_binary = BatchTable['SampleType'].isin(plot)
    x = x[plot_binary == True]
    t = t[plot_binary == True]
    z = z[plot_binary == True]
    #f = np.array(f)
    #f = f[plot_binary == True]
    sampletype = sampletype[plot_binary == True]
    order = BatchTable.Order[plot_binary == True]

    # Create empty grid (2x2)
    grid = np.full((2, 2), None)

    # Set y_label
    if transform is 'log':
        y_label = 'log(Peak Area)'
    else:
        y_label = 'Peak Area'

    # Get colors
    color_sampletype = BatchTable.SampleType[plot_binary == True].values
    col = []
    for i in range(len(color_sampletype)):
        if color_sampletype[i] == 'Blank':
            col.append('#00FF00')
        elif color_sampletype[i] == 'Sample':
            b_col = b[i]
            colmap = plt.get_cmap(colormap)
            b_rgb = colmap([b_col])
            b_hex = matplotlib.colors.rgb2hex(b_rgb[0])
            col.append(b_hex)
        elif color_sampletype[i] == 'QC':
            col.append('#FF0000')
        else:
            pass

    # Before correction plot
    grid[0, 1] = figure(title="Batch {} {}:{}".format(batch, PeakTable.Name[index], PeakTable.Label[index]), plot_width=600, plot_height=260, x_axis_label='Order', y_axis_label=y_label)
    grid[0, 1].title.text_font_size = '14pt'

    # Before: Plot line ('X' and dash)
    if gamma != False:
        source_before_line = ColumnDataSource(dict(x=t.values, y=f))
        glyph_before_x = X(x="x", y="y", line_width=2, fill_color=None)
        #glyph_before_line = Line(x="x", y="y", line_width=2, line_dash="dashed")
        grid[0, 1].add_glyph(source_before_line, glyph_before_x)
        #grid[0, 1].add_glyph(source_before_line, glyph_before_line)
    else:
        source_before_line = ColumnDataSource(dict(x=t.values, y=np.ones(len(t)) * x_qc_mean))
        glyph_before_line = Line(x="x", y="y", line_width=2, line_dash="dashed")
        grid[0, 1].add_glyph(source_before_line, glyph_before_line)

    # # Before: Plot circles
    source_before_circle = ColumnDataSource(dict(x=t.values, y=x.values, label=sampletype, color=col, Name=order))
    glyph_before_circle = grid[0, 1].circle(x="x", y="y", fill_color="color", fill_alpha=1, size=8, source=source_before_circle)

    # # Before: Add HoverTool
    grid[0, 1].add_tools(HoverTool(
        renderers=[glyph_before_circle],
        tooltips=[
            ("Type", "@label"),
            ("Order", "@Name"), ],))

    # # Before: Add control limit
    if control_limit == False:
        pass
    elif control_limit[0] in ['D-ratio', 'RSD']:
        if np.isnan(before_control_limit_low):  # Can't draw line if it doesn't exist
            pass
        else:
            before_control_limit_low = [before_control_limit_low] * len(t)
            before_control_limit_upp = [before_control_limit_upp] * len(t)
            source_before_control_limit = ColumnDataSource(dict(x=t.values, low=before_control_limit_low, upp=before_control_limit_upp))
            glyph_low = Line(x="x", y="low", line_width=2, line_dash="dashed", line_color='black')
            glyph_upp = Line(x="x", y="upp", line_width=2, line_dash="dashed", line_color='black')
            grid[0, 1].add_glyph(source_before_control_limit, glyph_low)
            grid[0, 1].add_glyph(source_before_control_limit, glyph_upp)

    if gamma != False:
        text_x = 72
    else:
        curvetype = 'nan'
        gamma_final = 'nan'
        text_x = 72

    # Textbox
    grid[0, 0] = figure(title="", plot_width=300, plot_height=265, x_axis_label="", y_axis_label="", outline_line_alpha=0)

    text1 = Label(x=text_x, y=210, x_units='screen', y_units='screen', text='Batch: {}'.format(batch), text_font_size='7.5pt')
    text2 = Label(x=text_x, y=190, x_units='screen', y_units='screen', text='Name: {}'.format(PeakTable.Name[index]), text_font_size='7.5pt')
    text3 = Label(x=text_x, y=170, x_units='screen', y_units='screen', text='Label: {}'.format(PeakTable.Label[index]), text_font_size='7.5pt')
    if transform is 'log':
        text4 = Label(x=text_x, y=150, x_units='screen', y_units='screen', text='log(MPA): {}'.format(np.round(mpa, 2)), text_font_size='7.5pt')
    else:
        text4 = Label(x=text_x, y=150, x_units='screen', y_units='screen', text='MPA: {}'.format(np.round(mpa, 2)), text_font_size='7.5pt')
    text5 = Label(x=text_x, y=130, x_units='screen', y_units='screen', text='Correction method: {}'.format(curvetype), text_font_size='7.5pt')
    text6 = Label(x=text_x, y=110, x_units='screen', y_units='screen', text='Optimal γ: {}'.format(gamma), text_font_size='7.5pt')
    text7 = Label(x=text_x, y=90, x_units='screen', y_units='screen', text='QC %RSD: {}'.format(np.round(Before_RSD_QC, 2)),
                  text_font_size='7.5pt')
    text8 = Label(x=text_x, y=70, x_units='screen', y_units='screen', text='Sam %RSD: {}'.format(np.round(Before_RSD_Sam, 2)),
                  text_font_size='7.5pt')
    text9 = Label(x=text_x, y=50, x_units='screen', y_units='screen', text='D-Ratio: {}'.format(np.round(Before_Dratio, 2)),
                  text_font_size='7.5pt')
    text10 = Label(x=text_x, y=30, x_units='screen', y_units='screen', text='Blank-Ratio: {}'.format(np.round(Before_BPAR, 2)),
                   text_font_size='7.5pt')

    grid[0, 0].add_layout(text1)
    grid[0, 0].add_layout(text2)
    grid[0, 0].add_layout(text3)
    grid[0, 0].add_layout(text4)
    grid[0, 0].add_layout(text5)
    grid[0, 0].add_layout(text6)
    grid[0, 0].add_layout(text7)
    grid[0, 0].add_layout(text8)
    grid[0, 0].add_layout(text9)
    grid[0, 0].add_layout(text10)
    grid[0, 0].circle(0, 0, line_color='white', fill_color='white', fill_alpha=0)  # Necessary to remove warning
    grid[0, 0].xaxis.visible = False
    grid[0, 0].yaxis.visible = False
    grid[0, 0].ygrid.visible = False
    grid[0, 0].xgrid.visible = False

    grid[0, 1].xgrid.visible = False
    grid[0, 1].ygrid.visible = False

    # Show figure
    fig = gridplot(grid.tolist())
    show(fig)
