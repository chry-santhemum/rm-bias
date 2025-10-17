import json
import plotly.graph_objects as go


def plot_bias_results(json_file_path: str, save_path: str | None = None):
    """
    Plot bias results from JSON file containing mean_results data.

    Args:
        json_file_path: Path to JSON file with mean_results structure
        save_path: Optional path to save the figure (e.g., 'plot.html' or 'plot.png')
    """
    with open(json_file_path, "r") as f:
        mean_results = json.load(f)

    attributes = list(mean_results.keys())

    # Extract scores for each condition
    plus_scores = [mean_results[attr]["plus"] for attr in attributes]
    minus_scores = [mean_results[attr]["minus"] for attr in attributes]
    original_scores = [mean_results[attr]["original"] for attr in attributes]

    # Create figure
    fig = go.Figure()

    # Add scatter plots for each condition
    fig.add_trace(
        go.Scatter(
            x=attributes,
            y=plus_scores,
            mode="markers",
            name="Plus",
            marker=dict(color="green", size=10),
            text=[
                f"{attr}<br>Plus: {score:.3f}"
                for attr, score in zip(attributes, plus_scores)
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=attributes,
            y=minus_scores,
            mode="markers",
            name="Minus",
            marker=dict(color="red", size=10),
            text=[
                f"{attr}<br>Minus: {score:.3f}"
                for attr, score in zip(attributes, minus_scores)
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=attributes,
            y=original_scores,
            mode="markers",
            name="Original",
            marker=dict(color="grey", size=10),
            text=[
                f"{attr}<br>Original: {score:.3f}"
                for attr, score in zip(attributes, original_scores)
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Update layout with legend outside the graph area
    fig.update_layout(
        title="Bias Analysis Results",
        xaxis_title="Attributes",
        yaxis_title="Score",
        xaxis=dict(tickangle=45),
        hovermode="closest",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.05),
    )

    # Save figure if path provided
    if save_path:
        (
            fig.write_html(save_path)
            if save_path.endswith(".html")
            else fig.write_image(save_path)
        )
        print(f"Figure saved to {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage
    plot_bias_results(
        "scrap/20251001-000007/rewrite_results_mean.json",
        save_path="scrap/20251001-000007/rewrite_results_mean.html",
    )
