# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bokeh>=3.3.0",
#     "pandas",
#     "fastmcp",
# ]
# ///
"""
SFA Viz - Visual Output Generator using Bokeh

This tool abstracts the complexity of generating interactive visualizations.
It produces standalone HTML files and opens them in the default browser.

Supported Types:
1. Histogram: Distribution of numerical data.
2. Sequence: UML-style sequence diagram from event logs.
3. Line: Time series or continuous data.
4. Bar: Categorical comparisons.
5. Scatter: Correlation between two variables.
6. Pie: Part-to-whole relationships.

Usage:
  uv run --script sfa_viz.py --type histogram --file data.csv --column "value"
  uv run --script sfa_viz.py --type sequence --file events.json
  uv run --script sfa_viz.py --type line --file data.csv --x "date" --y "value"
  uv run --script sfa_viz.py --type bar --file data.csv --x "category" --y "value"
  uv run --script sfa_viz.py --type scatter --file data.csv --x "height" --y "weight"
  uv run --script sfa_viz.py --type pie --file data.csv --x "category" --y "value"
"""

import argparse
import json
import sys
import webbrowser
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from fastmcp import FastMCP
    mcp = FastMCP("sfa-viz")
except ImportError:
    mcp = None

import pandas as pd
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import ColumnDataSource, HoverTool, Arrow, NormalHead, LabelSet
from bokeh.layouts import column
from bokeh.palettes import Spectral6, Category10
from bokeh.transform import cumsum, dodge

# --- Global Security ---
ALLOWED_PATHS: List[Path] = []

def _normalize_path(path_str: str) -> Path:
    if not path_str:
        return Path.cwd()
    path = Path(path_str).resolve()
    return path

def _load_data(file_path: Path) -> Any:
    """Load data from CSV, JSON, or TSV"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = file_path.suffix.lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.tsv':
        return pd.read_csv(file_path, sep='\t')
    elif ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

# --- Visualization Logic ---

def render_histogram(data: pd.DataFrame, col_name: str, title: str = "Histogram", output_path: str = "histogram.html", show_plot: bool = True):
    """Generates a Histogram from a DataFrame column"""
    if col_name not in data.columns:
        raise ValueError(f"Column '{col_name}' not found in data. Available: {list(data.columns)}")

    # Clean data: drop NaNs and ensure numeric
    clean_data = pd.to_numeric(data[col_name], errors='coerce').dropna()

    p = figure(title=title, x_axis_label=col_name, y_axis_label='Count', 
               tools="pan,wheel_zoom,box_zoom,reset,save",
               background_fill_color="#fafafa")

    # Calculate histogram
    import numpy as np
    hist, edges = np.histogram(clean_data, bins=20)
    
    # Create a dataframe for the quad glyphs to support hover tool
    hist_df = pd.DataFrame({
        'top': hist, 
        'left': edges[:-1], 
        'right': edges[1:],
        'count': hist,
        'interval': [f"{l:.2f} - {r:.2f}" for l, r in zip(edges[:-1], edges[1:])]
    })
    
    source = ColumnDataSource(hist_df)

    p.quad(top='top', bottom=0, left='left', right='right',
           fill_color="navy", line_color="white", alpha=0.5,
           source=source)

    # Add HoverTool
    hover = HoverTool(tooltips=[
        ("Interval", "@interval"),
        ("Count", "@count")
    ])
    p.add_tools(hover)

    output_file(output_path, title=title)
    if show_plot:
        show(p)
    else:
        save(p)
    
    return str(Path(output_path).absolute())

def render_sequence_diagram(events: List[Dict[str, Any]], title: str = "Sequence Diagram", output_path: str = "sequence.html", show_plot: bool = True):
    """
    Generates a Sequence Diagram from a list of event dictionaries.
    Expected format: [{'from': 'A', 'to': 'B', 'label': 'msg'}, ...]
    """
    # 1. Identify Actors
    actors = []
    for e in events:
        if e.get('from') and e['from'] not in actors:
            actors.append(e['from'])
        if e.get('to') and e['to'] not in actors:
            actors.append(e['to'])
    
    if not actors:
        raise ValueError("No actors found in event data.")

    # Map actors to X coordinates
    actor_map = {actor: i for i, actor in enumerate(actors)}
    
    # 2. Setup Figure
    # Y-axis is inverted (time goes down), so we use negative numbers
    num_events = len(events)
    p = figure(title=title, x_range=actors, y_range=(-num_events - 1, 1),
               height=max(400, num_events * 50 + 100),
               width=max(600, len(actors) * 150),
               tools="pan,wheel_zoom,save,reset",
               x_axis_location="above")
    
    p.yaxis.visible = False
    p.grid.grid_line_color = None

    # 3. Draw Lifelines (Vertical dashed lines)
    p.segment(x0=[i for i in range(len(actors))], y0=0,
              x1=[i for i in range(len(actors))], y1=-num_events,
              line_color="gray", line_dash="dashed", line_width=2)

    # 4. Draw Messages (Arrows)
    for i, event in enumerate(events):
        y_pos = -i
        start_actor = event.get('from')
        end_actor = event.get('to')
        label = event.get('label', '')
        
        if start_actor and end_actor and start_actor in actor_map and end_actor in actor_map:
            x_start = actor_map[start_actor]
            x_end = actor_map[end_actor]
            
            # Draw Arrow
            p.add_layout(Arrow(end=NormalHead(fill_color="orange", size=10),
                               x_start=x_start, y_start=y_pos,
                               x_end=x_end, y_end=y_pos,
                               line_color="orange", line_width=2))
            
            # Draw Label (centered above arrow)
            x_mid = (x_start + x_end) / 2
            p.text(x=[x_mid], y=[y_pos], text=[label], 
                   text_align="center", text_baseline="bottom",
                   text_font_size="10pt", y_offset=-2)

    output_file(output_path, title=title)
    if show_plot:
        show(p)
    else:
        save(p)

    return str(Path(output_path).absolute())

def render_line_chart(data: pd.DataFrame, x_col: str, y_cols: str, title: str = "Line Chart", output_path: str = "line.html", show_plot: bool = True):
    """Generates a Line Chart"""
    y_column_list = [c.strip() for c in y_cols.split(',')]
    
    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found. Available: {list(data.columns)}")
    
    for col in y_column_list:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(data.columns)}")

    # Try to parse dates if x_col looks like a date
    try:
        data[x_col] = pd.to_datetime(data[x_col])
        x_axis_type = "datetime"
    except Exception:
        x_axis_type = "auto"

    source = ColumnDataSource(data)
    
    p = figure(title=title, x_axis_label=x_col, y_axis_label="Value", 
               x_axis_type=x_axis_type,
               tools="pan,wheel_zoom,box_zoom,reset,save",
               background_fill_color="#fafafa")

    colors = Category10[10]
    
    for i, col_name in enumerate(y_column_list):
        color = colors[i % 10]
        p.line(x=x_col, y=col_name, source=source, line_width=2, color=color, legend_label=col_name)
        p.scatter(x=x_col, y=col_name, source=source, size=8, color=color, fill_color="white", legend_label=col_name)

    hover_tooltips = [(x_col, f"@{x_col}{{%F}}" if x_axis_type == "datetime" else f"@{x_col}")]
    for col_name in y_column_list:
        hover_tooltips.append((col_name, f"@{col_name}"))

    hover = HoverTool(tooltips=hover_tooltips, 
                      formatters={f'@{x_col}': 'datetime'} if x_axis_type == "datetime" else {})
    
    p.add_tools(hover)
    p.legend.click_policy = "hide"

    output_file(output_path, title=title)
    if show_plot:
        show(p)
    else:
        save(p)
    
    return str(Path(output_path).absolute())

def render_bar_chart(data: pd.DataFrame, x_col: str, y_cols: str, stacked: bool = False, title: str = "Bar Chart", output_path: str = "bar.html", show_plot: bool = True):
    """Generates a Bar Chart"""
    y_column_list = [c.strip() for c in y_cols.split(',')]

    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found. Available: {list(data.columns)}")
    
    for col in y_column_list:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(data.columns)}")

    # Ensure x_col is string for categorical axis
    data[x_col] = data[x_col].astype(str)
    categories = data[x_col].unique().tolist()

    source = ColumnDataSource(data)
    
    p = figure(title=title, x_range=categories, x_axis_label=x_col, y_axis_label="Value",
               tools="pan,wheel_zoom,box_zoom,reset,save",
               background_fill_color="#fafafa")

    colors = Category10[10]

    if stacked and len(y_column_list) > 1:
        p.vbar_stack(y_column_list, x=x_col, width=0.9, source=source, 
                     color=colors[:len(y_column_list)], legend_label=y_column_list)
    elif len(y_column_list) > 1:
        # Grouped (Dodged)
        width = 0.8 / len(y_column_list)
        start_offset = -0.4 + (width / 2)
        
        for i, col_name in enumerate(y_column_list):
            offset = start_offset + (i * width)
            p.vbar(x=dodge(x_col, offset, range=p.x_range), top=col_name, width=width, source=source,
                   color=colors[i % 10], legend_label=col_name)
    else:
        # Single Bar
        p.vbar(x=x_col, top=y_column_list[0], width=0.9, source=source, 
               line_color="white", fill_color="navy")

    hover_tooltips = [(x_col, f"@{x_col}")]
    for col_name in y_column_list:
        hover_tooltips.append((col_name, f"@{col_name}"))

    hover = HoverTool(tooltips=hover_tooltips)
    p.add_tools(hover)
    
    if len(y_column_list) > 1:
        p.legend.click_policy = "hide"

    output_file(output_path, title=title)
    if show_plot:
        show(p)
    else:
        save(p)
    
    return str(Path(output_path).absolute())

def render_scatter_plot(data: pd.DataFrame, x_col: str, y_cols: str, title: str = "Scatter Plot", output_path: str = "scatter.html", show_plot: bool = True):
    """Generates a Scatter Plot"""
    y_column_list = [c.strip() for c in y_cols.split(',')]

    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found. Available: {list(data.columns)}")
    
    for col in y_column_list:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(data.columns)}")

    source = ColumnDataSource(data)
    
    p = figure(title=title, x_axis_label=x_col, y_axis_label="Value",
               tools="pan,wheel_zoom,box_zoom,reset,save",
               background_fill_color="#fafafa")

    colors = Category10[10]

    for i, col_name in enumerate(y_column_list):
        p.scatter(x=x_col, y=col_name, source=source, size=10, 
                  color=colors[i % 10], alpha=0.5, legend_label=col_name)

    hover_tooltips = [(x_col, f"@{x_col}")]
    for col_name in y_column_list:
        hover_tooltips.append((col_name, f"@{col_name}"))

    hover = HoverTool(tooltips=hover_tooltips)
    p.add_tools(hover)
    p.legend.click_policy = "hide"

    output_file(output_path, title=title)
    if show_plot:
        show(p)
    else:
        save(p)
    
    return str(Path(output_path).absolute())

def render_pie_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str = "Pie Chart", output_path: str = "pie.html", show_plot: bool = True):
    """Generates a Pie Chart"""
    if x_col not in data.columns or y_col not in data.columns:
        raise ValueError(f"Columns '{x_col}' or '{y_col}' not found. Available: {list(data.columns)}")

    # Aggregate data if there are duplicate categories
    data = data.groupby(x_col)[y_col].sum().reset_index()
    
    data['angle'] = data[y_col] / data[y_col].sum() * 2 * math.pi
    
    # Assign colors
    from bokeh.palettes import Category20c
    if len(data) > 20:
        # Cycle colors if more than 20 categories
        palette = Category20c[20] * (len(data) // 20 + 1)
        data['color'] = palette[:len(data)]
    elif len(data) < 3:
         data['color'] = Category20c[3][:len(data)]
    else:
        data['color'] = Category20c[len(data)]

    source = ColumnDataSource(data)

    p = figure(title=title, toolbar_location=None,
               tools="hover", tooltips=f"@{x_col}: @{y_col}", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field=x_col, source=source)

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None

    output_file(output_path, title=title)
    if show_plot:
        show(p)
    else:
        save(p)
    
    return str(Path(output_path).absolute())

# --- Main Execution ---

def generate_viz(type: str, file_path: str, column: Optional[str] = None, x: Optional[str] = None, y: Optional[str] = None, output: Optional[str] = None, no_show: bool = False, stacked: bool = False):
    """Main logic to dispatch visualization generation"""
    path = _normalize_path(file_path)
    data = _load_data(path)
    
    if not output:
        output = f"{type}_output.html"

    if type == "histogram":
        if not column:
            # Try to guess a numeric column if not provided
            if isinstance(data, pd.DataFrame):
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    column = numeric_cols[0]
                    print(f"Auto-selected column: {column}")
                else:
                    raise ValueError("No numeric column found and none specified.")
            else:
                 raise ValueError("Data must be tabular (CSV/JSON) for histogram.")
        
        return render_histogram(data, column, output_path=output, show_plot=not no_show)

    elif type == "sequence":
        # Ensure data is a list of dicts
        if isinstance(data, pd.DataFrame):
            events = data.to_dict(orient='records')
        elif isinstance(data, list):
            events = data
        else:
            raise ValueError("Data must be a list of events for sequence diagram.")
            
        return render_sequence_diagram(events, output_path=output, show_plot=not no_show)

    elif type == "line":
        if not x or not y:
             raise ValueError("Line chart requires --x and --y columns.")
        return render_line_chart(data, x, y, output_path=output, show_plot=not no_show)

    elif type == "bar":
        if not x or not y:
             raise ValueError("Bar chart requires --x and --y columns.")
        return render_bar_chart(data, x, y, stacked=stacked, output_path=output, show_plot=not no_show)

    elif type == "scatter":
        if not x or not y:
             raise ValueError("Scatter plot requires --x and --y columns.")
        return render_scatter_plot(data, x, y, output_path=output, show_plot=not no_show)

    elif type == "pie":
        if not x or not y:
             raise ValueError("Pie chart requires --x (category) and --y (value) columns.")
        return render_pie_chart(data, x, y, output_path=output, show_plot=not no_show)

    else:
        raise ValueError(f"Unknown visualization type: {type}")

# --- MCP Tools ---

if mcp:
    @mcp.tool()
    def create_histogram(file_path: str, column: str, output_path: str = "histogram.html") -> str:
        """Create a histogram from a data file."""
        try:
            result = generate_viz("histogram", file_path, column=column, output=output_path, no_show=True)
            return f"Histogram created at: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def create_sequence_diagram(file_path: str, output_path: str = "sequence.html") -> str:
        """Create a sequence diagram from a JSON/CSV file of events."""
        try:
            result = generate_viz("sequence", file_path, output=output_path, no_show=True)
            return f"Sequence diagram created at: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def create_line_chart(file_path: str, x: str, y: str, output_path: str = "line.html") -> str:
        """Create a line chart from a data file."""
        try:
            result = generate_viz("line", file_path, x=x, y=y, output=output_path, no_show=True)
            return f"Line chart created at: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def create_bar_chart(file_path: str, x: str, y: str, output_path: str = "bar.html") -> str:
        """Create a bar chart from a data file."""
        try:
            result = generate_viz("bar", file_path, x=x, y=y, output=output_path, no_show=True)
            return f"Bar chart created at: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def create_scatter_plot(file_path: str, x: str, y: str, output_path: str = "scatter.html") -> str:
        """Create a scatter plot from a data file."""
        try:
            result = generate_viz("scatter", file_path, x=x, y=y, output=output_path, no_show=True)
            return f"Scatter plot created at: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    def create_pie_chart(file_path: str, x: str, y: str, output_path: str = "pie.html") -> str:
        """Create a pie chart from a data file."""
        try:
            result = generate_viz("pie", file_path, x=x, y=y, output=output_path, no_show=True)
            return f"Pie chart created at: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="SFA Viz - Visual Output Generator")
    parser.add_argument("--type", choices=["histogram", "sequence", "line", "bar", "scatter", "pie"], required=True, help="Type of visualization")
    parser.add_argument("--file", required=True, help="Path to data file (CSV, JSON)")
    parser.add_argument("--column", help="Column name for histogram")
    parser.add_argument("--x", help="X-axis column name")
    parser.add_argument("--y", help="Y-axis column name(s), comma-separated")
    parser.add_argument("--stacked", action="store_true", help="Stack bars in bar chart")
    parser.add_argument("--output", help="Output HTML file path")
    parser.add_argument("--no-show", action="store_true", help="Do not open browser automatically")
    
    args = parser.parse_args()
    
    try:
        # Pass stacked argument only if type is bar, but generate_viz needs to handle it or we pass it via kwargs
        # For simplicity, let's update generate_viz signature or just pass it if type is bar
        if args.type == "bar":
             output = generate_viz(args.type, args.file, args.column, args.x, args.y, args.output, args.no_show, stacked=args.stacked)
        else:
             output = generate_viz(args.type, args.file, args.column, args.x, args.y, args.output, args.no_show)
             
        print(json.dumps({"success": True, "output": output}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    if mcp and len(sys.argv) == 1:
        mcp.run()
    else:
        main()
