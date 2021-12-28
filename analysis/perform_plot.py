import os
from itertools import product
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils import get_project_root


def combine_plots(file_names, saved_path):
    combined_image = Image.new('RGBA', (3000, 1000))
    for i in range(2):
        image = Image.open(file_names[i])
        # loc = ((i % 2) * 200, (int(i/2) * 200))
        loc = (((i % 2) * 1400 + 200 * i), int(i / 2) * 1400)
        combined_image.paste(image, loc)

    combined_image.save(saved_path)


def plot_one(df, name, ranges, img_file, acc, macro):
    baselines = {
        "News26":
            [
                {"y": 72.31, "annotation_text": "BiLSTM+Att(72.31)",
                 "line_color": "black", "line_dash": "dot", "annotation_font_color": "black"},
                {"y": 71.4, "annotation_text": "GRU+Att(71.4)"},
                {"y": 69.36, "annotation_text": "NRMS news encoder(69.36)"},
                {"y": 68.99, "annotation_text": "Fastformer(68.99)", "annotation_position": "bottom right",
                 "line_color": "black", "line_dash": "dash", "annotation_font_color": "black"},
            ],
        "MIND15":
            [
                {"y": 79.12, "annotation_text": "BiLSTM+Att(79.12)"},
                {"y": 78.77, "annotation_text": "GRU+Att(78.77)", "annotation_position": "bottom right",
                 "line_color": "black", "line_dash": "dash", "annotation_font_color": "black"},
                {"y": 78.83, "annotation_text": "NRMS news encoder(78.83)"},
                {"y": 79.78, "annotation_text": "Fastformer(79.78)",
                 "line_color": "black", "line_dash": "dot", "annotation_font_color": "black"},
            ]
    }
    head_num = sorted(df.head_num.values)
    accuracy = [float(df.loc[df["head_num"] == n, acc].values[0].split(u"\u00B1")[0]) for n in head_num]
    macro_f = [float(df.loc[df["head_num"] == n, macro].values[0].split(u"\u00B1")[0]) for n in head_num]
    head_num = [f"Head-{n}" for n in head_num]
    # Create figure with secondary y-axis
    subplot = make_subplots(specs=[[{"secondary_y": True}]])
    # subplot = go.Figure(layout_yaxis_range=yrange)
    subplot.add_trace(go.Bar(x=head_num, y=accuracy, name='Accuracy', text=np.round(accuracy, 2), textposition="outside",
                             marker_pattern_shape="x", yaxis='y', offsetgroup=1), secondary_y=False)
    subplot.add_trace(go.Bar(x=head_num, y=macro_f, name='Macro-F', text=np.round(macro_f, 2), textposition="outside",
                             marker_pattern_shape="/", yaxis='y2', offsetgroup=2), secondary_y=True)
    for line in baselines[name]:
        subplot.add_hline(**line)
    subplot.update_layout(autosize=True, margin={'l': 0, 'r': 0, 't': 10})
    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    subplot.update_layout(barmode='group', xaxis_tickangle=0, yaxis_title="Performance",
                          xaxis_title="The number of attention heads")
    subplot.update_layout(legend=dict(x=0.05, yanchor="top", xanchor="left"))
    subplot.update_layout(plot_bgcolor="white", legend_font_size=12, font_size=12, font_color="black")
    subplot.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    subplot.update_yaxes(showgrid=True, gridcolor="lightgrey", showline=True, linewidth=1, linecolor='black', mirror=True)
    subplot.update_yaxes(title_text="Accuracy", secondary_y=False, range=ranges[0])
    subplot.update_yaxes(title_text="Macro-F", secondary_y=True, range=ranges[1])
    subplot.update_layout(title={"text": name, "x": 0.5, "y": 0.01, 'xanchor': 'center', 'yanchor': 'bottom'})
    subplot.write_image(img_file, scale=2)
    return subplot


if __name__ == "__main__":
    names = ["News26", "MIND15"]
    datasets = ["keep_all", "aggressive"]
    test_args = ["head_num"]
    acc_range = {"MIND15": [76, 82], "News26": [66, 73]}
    macro_range = {"MIND15": [54, 66], "News26": [50, 62]}
    sets = [("val_accuracy", "val_macro_f")]
    root_path = Path(get_project_root()) / "saved"
    image_root = root_path / "plot"
    for name, set_type, arg, ss in product(names, datasets, test_args, sets):
        os.makedirs(image_root, exist_ok=True)
        file_name = f"{name}_{set_type}_{arg}"
        file_path = root_path / "stat" / f"{file_name}.csv"
        if os.path.exists(file_path):
            stat_df = pd.read_csv(file_path)
            for (vn, at), group in stat_df.groupby(["variant_name", "arch_type"]):
                image_path = image_root / f"{name}_{vn}_{ss[0].split('_')[0]}.png"
                fig1 = plot_one(group, name, (acc_range[name], macro_range[name]), image_path, ss[0], ss[1])
    variant_names = ["raw", "combined_mha", "reuse", "topic_embed", "combined_gru", "weight_mha"]
    arch_types = ["BiAttentionClassifyModel"]
    for set_type, arg, vn, at, ss in product(datasets, test_args, variant_names, arch_types, sets):
        mind_image = image_root / f"MIND15_{vn}_{ss[0].split('_')[0]}.png"
        news_image = image_root / f"News26_{vn}_{ss[0].split('_')[0]}.png"
        saved_path = image_root / f"MIND15_News26_{vn}_{ss[0].split('_')[0]}.png"
        if os.path.exists(mind_image) and os.path.exists(news_image):
            combine_plots([mind_image, news_image], saved_path)
