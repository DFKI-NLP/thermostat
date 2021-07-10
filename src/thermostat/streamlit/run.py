import datasets
import glob
import json
import pandas as pd
import streamlit as st
import sys
import textwrap

from thermostat import load
from thermostat.data.thermostat_configs import builder_configs


nlp = datasets

HTML_WRAPPER = """<div>{}</div>"""
#HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem;
#                  margin-bottom: 2.5rem">{}</div>"""
MAX_SIZE = 40000000000
if len(sys.argv) > 1:
    path_to_datasets = sys.argv[1]
else:
    path_to_datasets = None


# Hack to extend the width of the main pane.
def _max_width_():
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    th {{
        text-align: left;
        font-size: 110%;


     }}

    tr:hover {{
        background-color: #ffff99;
        }}

    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()


def render_features(features):
    if isinstance(features, dict):
        return {k: render_features(v) for k, v in features.items()}
    if isinstance(features, nlp.features.ClassLabel):
        return features.names

    if isinstance(features, nlp.features.Value):
        return features.dtype

    if isinstance(features, nlp.features.Sequence):
        return {"[]": render_features(features.feature)}
    return features


app_state = st.experimental_get_query_params()
start = True
loaded = True
INITIAL_SELECTION = ""

app_state.setdefault("dataset", "glue")
if len(app_state.get("dataset", [])) == 1:
    app_state["dataset"] = app_state["dataset"][0]
    INITIAL_SELECTION = app_state["dataset"]
#print(INITIAL_SELECTION)

if start:
    # Logo and sidebar decoration.
    st.sidebar.markdown(
        """<center>
    <a href="https://github.com/DFKI-NLP/thermostat">
    </a>
    </center>""",
        unsafe_allow_html=True,
    )
    st.sidebar.image("logo.png", width=300)
    st.sidebar.markdown(
        "<center><h2><a href='https://github.com/DFKI-NLP/thermostat'>github/DFKI-NLP/thermostat</h2></a></center>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
    <center>
        <a target="_blank" href="https://huggingface.co/docs/datasets/">datasets Docs</a>
    </center>""",
        unsafe_allow_html=True,
    )
    st.sidebar.subheader("")

    # Interaction with the nlp libary.
    # @st.cache
    def get_confs():
        """  Get the list of confs for a dataset. """
        confs = builder_configs
        if confs and len(confs) > 1:
            return confs
        else:
            return []

    # @st.cache(allow_output_mutation=True)
    def get(conf):
        """ Get a dataset from name and conf """
        ds = load(conf, cache_dir=path_to_datasets)
        return ds, False

    # Dataset select box.
    datasets = []
    selection = None

    if path_to_datasets is None:
        list_of_datasets = nlp.list_datasets(with_community_datasets=False)
    else:
        list_of_datasets = sorted(glob.glob(path_to_datasets + "*"))
    for i, dataset in enumerate(list_of_datasets):
        dataset = dataset.split("/")[-1]
        if INITIAL_SELECTION and dataset == INITIAL_SELECTION:
            selection = i
        datasets.append(dataset)

    st.experimental_set_query_params(**app_state)

    # Side bar Configurations.
    configs = get_confs()
    conf_avail = len(configs) > 0
    conf_option = None
    if conf_avail:
        start = 0
        for i, conf in enumerate(configs):
            if conf.name == app_state.get("config", None):
                start = i
        conf_option = st.sidebar.selectbox(
            "Thermostat configuration", configs, index=start, format_func=lambda a: a.name
        )
        app_state["config"] = conf_option.name

    else:
        if "config" in app_state:
            del app_state["config"]
    st.experimental_set_query_params(**app_state)

    dts, fail = get(str(conf_option.name) if conf_option else None)

    # Main panel setup.
    if fail:
        st.markdown(
            "Dataset is too large to browse or requires manual download. Check it out in the datasets library! \n\n "
            "Size: "
            + str(dts.info.size_in_bytes)
            + "\n\n Instructions: "
            + str(dts.manual_download_instructions)
        )
    else:
        d = dts
        keys = list(d[0].__dict__.keys())

        st.header(
            "Thermostat configuration: "
            + (conf_option.name if conf_option else "")
        )

        st.markdown(
            "*Homepage*: "
            + d.info.homepage
        )

        md = """
        %s
        """ % (
            d.info.description.replace("\\", "  ")
        )
        st.markdown(md)

        step = 50
        offset = st.sidebar.number_input(
            "Offset (Size: %d)" % len(d),
            min_value=0,
            max_value=int(len(d)) - step,
            value=0,
            step=step,
        )

        citation = None #st.sidebar.checkbox("Show Citations", False)
        table = not st.sidebar.checkbox("Show List View", False)
        show_features = st.sidebar.checkbox("Show Features", True)
        show_atts = st.sidebar.checkbox("Show Attribution Scores", False)
        md = """
```
%s
```
""" % (
            d.info.citation.replace("\\", "").replace("}", " }").replace("{", "{ "),
        )
        if citation:
            st.markdown(md)
        # st.text("Features:")
        #if show_features:
        #    on_keys = st.multiselect("Features", keys, keys)
        #    #st.write(render_features(d.features))
        #else:
        on_keys = keys

        # Remove some keys
        on_keys = [k for k in on_keys if k in ['predictions', 'true_label', 'predicted_label']]

        if not table:
            # Full view.
            for item in range(offset, offset + step):
                st.text("        ")
                st.text("                  ----  #" + str(item))
                st.text("        ")
                # Use st to write out.
                for k in on_keys:
                    v = getattr(d[item], k)
                    st.subheader(k)
                    if isinstance(v, str):
                        out = v
                        st.text(textwrap.fill(out, width=120))
                    elif (
                            isinstance(v, bool)
                            or isinstance(v, int)
                            or isinstance(v, float)
                    ):
                        st.text(v)
                    else:
                        st.write(v)

        else:
            # Table view. Use Pandas.
            df, heatmap_htmls = [], []
            for item in range(offset, offset + step):
                df_item = {}
                df_item["_number"] = item
                for k in on_keys:
                    v = getattr(d[item], k)

                    # Remove [PAD] tokens from attributions and input_ids
                    if k in ['attributions', 'input_ids']:
                        v = [vi for vi in v if vi != 0 or vi != 0.0]

                    if isinstance(v, str):
                        out = v
                        df_item[k] = textwrap.fill(out, width=50)
                    elif (
                            isinstance(v, bool)
                            or isinstance(v, int)
                            or isinstance(v, float)
                    ):
                        df_item[k] = v
                    else:
                        out = json.dumps(v, indent=2, sort_keys=True)
                        df_item[k] = out

                # Add heatmap viz
                html = getattr(d[item], 'heatmap').render(labels=show_atts)
                html = html.replace("\n", " ")
                heatmap_htmls.append(HTML_WRAPPER.format(html))

                df.append(df_item)
            df2 = df
            df = pd.DataFrame(df).set_index("_number")

            def hover(hover_color="#ffff99"):
                return dict(
                    selector="tr:hover",
                    props=[("background-color", "%s" % hover_color)],
                )

            styles = [
                hover(),
                dict(
                    selector="th",
                    props=[("font-size", "150%"), ("text-align", "center")],
                ),
                dict(selector="caption", props=[("caption-side", "bottom")]),
            ]
            # Table view. Use pands styling.
            style = df.style.set_properties(
                **{"text-align": "left", "white-space": "pre"}
            ).set_table_styles([dict(selector="th", props=[("text-align", "left")])])
            style = style.set_table_styles(styles)  # Setting the style appears to be broken for streamlit+pandas

            for i, heatmap_html in enumerate(heatmap_htmls):
                st.write(HTML_WRAPPER.format(heatmap_html), unsafe_allow_html=True)
                st.table(df.iloc[[i]])
                st.markdown(""" --- """)

    # Additional dataset installation and sidebar properties.
    md = """
    ### Code

    ```python
    !pip install thermostat_datasets
    from thermostat import load
    dataset = load(
       '%s)
    ```

    """ % (
        (conf_option.name + "'") if conf_option else "",
    )
    st.sidebar.markdown(md)
