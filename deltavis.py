import numpy as np
import torch as t
from einops import rearrange, pack
import fire
from bokeh.io import output_notebook 
from bokeh.models.tools import WheelZoomTool
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256

def main(p):
    # b d l
    template = 'delta.{}.{}.pt'

    delta0 = t.load(template.format(0, p))
    delta1 = t.load(template.format(1, p))
    delta, ps = pack([delta0, delta1], 'b * l')
    delta = delta[:3]
    delta = rearrange(delta, 'b (s d) l -> (b d s) l', s=2).numpy()

    # print(delta.shape)
    # return

    # Prepare data
    Y, X = delta.shape
    x = np.arange(0, X, 1).astype('float64')
    y = np.arange(0, Y, 1).astype('float64')
    xx, yy = np.meshgrid(x, y)
    yy += (yy // 2) * 0.5 + (yy // 64) * 3

    # Flatten the arrays for ColumnDataSource
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    color_flat = delta.flatten()

    # Create a ColumnDataSource
    source = ColumnDataSource(data=dict(x=xx_flat, y=yy_flat, color=color_flat))

    title = 'Softplus Delta values in Induction task (d_model = 16)'
    # Create a figure
    p = figure(title=title, 
               x_range=(-0.5, xx.max()+0.5), 
               y_range=(-0.5, yy.max()+0.5),
               # tools="wheel_zoom,box_zoom,hover", 
               tooltips=[('x', '@x'), ('y', '@y'), ('value', '@color')],
               width=1800, height=2500)
    p.xaxis.axis_label = "Timestep"
    p.yaxis.axis_label = "d_inner (group) x 10 samples per group"
    p.xaxis.axis_label_text_font_size = "24pt"
    p.yaxis.axis_label_text_font_size = "24pt" 
    p.title.text_font_size = '24pt'
    # Map colors
    mapper = linear_cmap(field_name='color', palette=Viridis256, low=min(color_flat),
                         high=4.0)
                         # high=max(color_flat))

    # Add rectangles
    p.rect(x="x", y="y", width=0.99, height=1, source=source, line_color=None,
           fill_color=mapper)

    # Add color bar
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
    p.add_layout(color_bar, 'right')

    # Output the plot
    # output_notebook()  # For Jupyter Notebook
    show(p)  # Display the plot

if __name__ == '__main__':
    fire.Fire(main)

