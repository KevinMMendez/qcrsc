import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from bokeh.plotting import ColumnDataSource, figure, show, output_notebook
from bokeh.models import Label, HoverTool, Span, Range1d
from bokeh.layouts import gridplot, column
from sklearn.preprocessing import minmax_scale


def scatter_plot(PeakTable, x, y, label='Name', title='Plot', size=10, size_min=10, size_max=100, color='red', color_scale='test', axis_type=('linear', 'linear'), width=500, height=500, alpha=0.5, legend=False, hline=0, vline=0, colormap='plasma', x_range=False, y_range=False):

  x_score = PeakTable[x]
  y_score = PeakTable[y]

  # convert x and y to numpy array
  x_score = np.array(x_score)
  y_score = np.array(y_score)

  # if size is string, get peaktable column
  if isinstance(size, str) == True:
    size_before = np.array(PeakTable[size]).reshape(1, -1)
    size_scaled = minmax_scale(size_before, feature_range=(size_min, size_max), axis=1)
    size_scaled = size_scaled[0]
  else:
    size_before = np.array([size] * len(x_score)).reshape(1, -1)
    size_scaled = [size] * len(x_score)
    size = 'size'

  # Color is a column in peaktable, or it is the color e.g. 'red'
  try:
    col_before = PeakTable[color].values
    if isinstance(col_before[0], str):
      col = col_before
    else:
      col_before = np.array(PeakTable[color]).reshape(1, -1)
      col_scaled = minmax_scale(col_before, feature_range=(0, 1), axis=1)[0]
      colmap = plt.get_cmap(colormap)
      col_rbg = colmap(col_scaled)
      col = []
      for i in col_rbg:
        col.append(matplotlib.colors.rgb2hex(i))
      col_name = PeakTable[color].values
  except KeyError:
    col = [color] * len(size_scaled)
    col_name = col

  data = dict(x=x_score, y=y_score, size=size_scaled, col=col, size_text=np.round(size_before, 2)[0])

  # Bokeh get labels
  data_label = {}
  if isinstance(label, list) == False:
    data_label[label] = PeakTable[label]
  else:
    for i in label:
      data_label[i] = PeakTable[i]

  data.update(data_label)
  source = ColumnDataSource(data=data)

  fig = figure(title=title,
               x_axis_label=x,
               y_axis_label=y,
               x_axis_type=axis_type[0],
               y_axis_type=axis_type[1],
               plot_height=height,
               plot_width=width)

  glyph_circle = fig.circle(x="x",
                            y="y",
                            line_color='white',
                            fill_color='col',
                            fill_alpha=alpha,
                            size="size",
                            source=source)

  if hline is not False:
    h = Span(location=hline, dimension="width", line_color="black", line_width=1, line_alpha=0.75)
    fig.add_layout(h)

  if vline is not False:
    v = Span(location=vline, dimension="height", line_color="black", line_width=1, line_alpha=0.75)
    fig.add_layout(v)

  # Tool-tip (add everything in label_copy)
  TOOLTIPS = []
  for name, val in data_label.items():
    TOOLTIPS.append((str(name), "@" + str(name)))
  TOOLTIPS.append((size, "@size_text"))

  fig.add_tools(HoverTool(
      renderers=[glyph_circle],
      tooltips=TOOLTIPS,))

  if x_range is not False:
    fig.x_range = Range1d(x_range[0], x_range[1])

  if y_range is not False:
    fig.y_range = Range1d(y_range[0], y_range[1])

  fig.xgrid.visible = False
  fig.ygrid.visible = False
  fig.title.text_font_size = '14pt'
  output_notebook()
  #fig.output_backend = 'svg'
  show(column(fig))
