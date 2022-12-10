from matplotlib.cm import get_cmap
import matplotlib as mpl
from matplotlib.patches import Patch
import pandas as pd

from skater.data.datamanager import DataManager as DM
from skater.util.exceptions import MatplotlibUnavailableError
from skater.util.logger import build_logger
from skater.util.logger import _INFO
from skater.core.local_interpretation.text_interpreter import relevance_wt_assigner
from skater.util.dataops import convert_dataframe_to_dict
from skater.util.text_ops import generate_word_list

logger = build_logger(_INFO, __name__)


def __set_plot_feature_relevance_keyword(**plot_kw):
    plot_name = plot_kw['plot_name'] if 'plot_name' in plot_kw.keys() else 'feature_relevance.png'
    top_k = plot_kw['top_k'] if 'top_k' in plot_kw.keys() else 10
    color_map = plot_kw['color_map'] if 'color_map' in plot_kw.keys() else ('Red', 'Blue')
    fig_size = plot_kw['fig_size'] if 'fig_size' in plot_kw.keys() else (20, 10)
    font_name = plot_kw['font_name'] if 'font_name' in plot_kw.keys() else "Avenir Black"
    txt_font_size = plot_kw['txt_font_size'] if 'txt_font_size' in plot_kw.keys() else '14'
    return plot_name, top_k, color_map, fig_size, font_name, txt_font_size


# Reference: https://stackoverflow.com/questions/30618002/static-variable-in-a-function-with-python-decorator
def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


# The counter is added for a weird caching error that occurs when including images to html
# https://stackoverflow.com/questions/1431512/is-there-a-way-to-force-browsers-to-refresh-download-images
@static_var("plot_counter", 0)
def build_visual_explainer(text, relevance_scores, font_size='12pt', file_name='rendered', title='Word Relevance',
                           pos_clr_name='Reds', neg_clr_name='Blues', alpha=0.7, highlight_oov=False,
                           enable_plot=False, **plot_kw):
    """
    Build a visual explainer highlighting the positive(color code: Red(default)) and
    negative(color code: Blue(default)) relevance of the input features

    Parameters
    ----------
    text: str
        input text to be explained
    relevance_scores: pandas.core.series.Series
        contribution/attribution scores for the features corresponding to the given text.
        The relevance scores are compute over the embedding vector, remember to aggregate the embedding dimension
        by computing 'mean/sum' to get scalar coefficient for the features.
    font_size: str (default: `12pt`)
        font size for the text visualizer
    file_name: str (default: 'rendered')
        specify the file name for the html file to be generated
    title: str
        specify a title
    pos_clr_name: str (default: 'Reds')
        color name for highlight positive influence. Color names should be compatible with matplotlib supported color names
        (https://matplotlib.org/gallery/color/named_colors.html)
    neg_clr_name: str (default: 'Blues')
        color name for highlight negative influence. Color names should be compatible with matplotlib supported color names
        (https://matplotlib.org/gallery/color/named_colors.html)
    alpha: int (default: 0.7)
        adjust the transparency for display
    highlight_oov: boolean (default: False)
        highlighting words which are not part of the dictionary
    enable_plot: boolean (default: False)
        if True, generate feature relevance plots as well
    plot_kw: optional
        optional aesthetic features for plotting,
        - plot_name: str (default: feature_relevance.png)
        - top_k: int (default: 10)
        - color_map: tuple of str('positive_color', 'negative_color') (default: ('Red', 'Blue'))
        - fig_size: tuple of int(width, height) (default: (20, 10)
        - font_name: str (default: 'Avenir Black')
        - txt_font_size: str( default: '14')

    Returns
    -------
     None

    """
    # TODO: Add support for better visualization and plotting frameworks- e.g bokeh, plotly
    # Process the raw text to a word list
    _words = generate_word_list(text, ' ')
    features_df = pd.DataFrame({'features': _words})
    scores_df = pd.DataFrame({'relevance_scores': relevance_scores.tolist()})
    # Merge the data-frame column-wise, assert if the length of the word list and relevance
    # score data-frame does not match. This is important as currently the join is done on the index
    assert len(_words) == scores_df.shape[0]
    words_scores_df = features_df.join(scores_df)
    # assign column names to df containing 'words | relevance score'
    words_scores_df.columns = ['features', 'relevance_scores']
    words_scores_dict = convert_dataframe_to_dict('features', 'relevance_scores', words_scores_df)

    # generate plot displaying feature relevance rank ordered
    f_name = plot_feature_relevance(words_scores_df, **plot_kw) if enable_plot else None
    logger.info("Relevance plot name: {}".format(f_name))

    # build the html structure
    h_str = _build_str(text, words_scores_dict, f_name, title, font_size, pos_clr_name,
                       neg_clr_name, alpha, highlight_oov)
    # build html
    _build_html_file(html_str=h_str, file_name=file_name)


def _build_str(text, words_scores_dict, plot_file_name, title, font_size,
               pos_clr_name, neg_clr_name, alpha_value, highlight_oov):
    """
    References
    ----------
    ..  [1] https://github.com/cod3licious/textcatvis/blob/master/textcatvis/vis_utils.py
    ..  [2] http://matplotlib.org/examples/color/colormaps_reference.html
    """
    # TODO: Add a HTML validator to verify that HTML structure
    # color-maps for the words
    cmap_pos = get_cmap(pos_clr_name)
    cmap_neg = get_cmap(neg_clr_name)
    # Highlight with 'yellow' if the word is not present in
    # the word dictionary or not contributing to actual prediction
    rgba = (1., 1., 0)
    norm = mpl.colors.Normalize(0., 1.)

    html_content = u'<body><h3>{}</h3>' \
                   u'<div class="row" style=background-color:#F5F5F5 ' \
                   u'white-space: pre-wrap; ' \
                   u'font-size: {}; ' \
                   u'font-family: Avenir Black>'.format(title, font_size)

    rest_text = text
    relevance_wts = relevance_wt_assigner(text, words_scores_dict)
    for word, wts in relevance_wts:
        html_content += rest_text[: rest_text.find(word)]
        # cut off the identified word
        rest_text = rest_text[rest_text.find(word) + len(word):]
        # adjust opacity for non-vocabulary words
        alpha = 0.5 if highlight_oov else 0.
        if wts is not None:
            # override the RGB values for color coding
            rgba = cmap_neg(norm(-wts)) if wts < 0 else cmap_pos(norm(wts))
            # adjusting opacity for in-dictionary words
            alpha = alpha_value
        html_content += u'<span style="background-color: rgba({:.0f}, {:.0f}, {:.0f}, {:.1f})">{}</span>'\
            .format(round(float(255) * rgba[0]), round(float(255) * rgba[1]), round(float(255) * rgba[2]), alpha, word)
    # rest of the text
    html_content += rest_text
    html_content += u'</div>'

    # Embed the feature relevance scores as a plot
    build_visual_explainer.plot_counter += 1
    html_content += u'<div align="center"><img src="./{}?{}"</div>'. \
        format(plot_file_name, build_visual_explainer.plot_counter) if plot_file_name is not None else ''
    html_content += u'</body>'
    return html_content


def _build_html_file(html_str, file_name):
    file_name_with_extension = '{}.html'.format(file_name)
    with open(file_name_with_extension, 'w', encoding='utf8') as f:
        f.write(html_str)
        logger.info("Visual Explainer built, "
                    "use show_in_notebook to render in Jupyter style Notebooks: {}".format(file_name_with_extension))


def plot_feature_relevance(feature_relevance_scores, **plot_kw):
    """ Plotting function for visualizing the feature contribution rank ordered wrt +ve/-ve effect

    Parameters
    ----------
    feature_relevance_scores: pandas.core.frame.DataFrame
    plot_kw: optional
        optional aesthetic features for plotting,
        - plot_name: str (default: feature_relevance.png)
        - top_k: int (default: 10)
        - color_map: tuple of str('positive_color', 'negative_color') (default: ('Red', 'Blue'))
        - fig_size: tuple of int(width, height) (default: (20, 10)
        - font_name: str (default: 'Avenir Black')
        - txt_font_size: str( default: '14')

    Returns
    -------
    f_name: str
        returns the generate plot saved as *.png file. Other formats could be specified as well.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on the system."))

    # Validate the input data format. Currently supported types include (pandas.DataFrame)
    DM._check_input(feature_relevance_scores)
    # set the params needed for the plot
    f_name, top_k, color_map, fig_size, font_name, txt_font_size = __set_plot_feature_relevance_keyword(**plot_kw)

    # set the colors
    pos_color = color_map[0]
    neg_color = color_map[1]

    df = feature_relevance_scores.sort_values(by='relevance_scores', ascending=False)
    df['positive'] = df['relevance_scores'] > 0

    # Setting the style globally for the plot
    # For other style check the reference here `plt.style.available`
    plt.clf()
    plt.style.use('bmh')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # filter the top k features wrt each signed category (Positive / Negative ) features
    df_filtered = df.groupby('positive').head(top_k)
    color_list = df_filtered['positive'].map({True: pos_color, False: neg_color})
    df_filtered['relevance_scores'].plot(ax=ax, kind='barh', color=color_list, figsize=fig_size, width=0.85)
    custom_lines = [Patch(facecolor=pos_color, edgecolor='r'), Patch(facecolor=neg_color, edgecolor='b')]

    ax.legend(custom_lines, ['positive', 'negative'], loc='best')
    ax.yaxis.set_visible(False)
    # Display the feature names on the plot
    for index, f_x in enumerate(df_filtered['features']):
        ax.text(0, index + 0.5, f_x, ha='right', fontsize=txt_font_size, fontname=font_name)
    plt.savefig(f_name)
    logger.info("Rank order feature relevance based on input created and saved as {}".format(f_name))
    return f_name
