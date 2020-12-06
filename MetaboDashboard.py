import pandas as pd
import numpy as np
import seaborn as sns
import io, base64, glob
from collections import Counter


import dash, dash_bio, dash_table
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
from dash.exceptions import PreventUpdate

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import umap

import plotly.graph_objs as go

from LearnConfig import *
from ExperimentDesign import *
from Utils import *
from MetaboDashboardConfig import *


app = dash.Dash("MetaboDashboard", meta_tags=[{"name": "viewport", "content": "width=device-width"}])
server = app.server
app.scripts.config.serve_locally = False
app.css.config.serve_locally = False
app.config.suppress_callback_exceptions = True
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})


def layout():
    return html.Div(id="page", children=[
        html.Div(id="dataCache", children=[

        ],
        style={"display": "none"}
        ),
        html.Div(id="title_container", className="row", children=[
            html.H1(id="title", children="MetaboDashboard - Mediterranean vs North-American Diet", style={"text-align": "center"})
        ]),
        html.Div(id="main-content", children=[
            html.Div(id="menu_content", className="two columns", children=[
                html.H6("Design"),
                dcc.Dropdown(id="design_dropdown",
                    options=[{"label": i, "value": i} for i in EXPERIMENT_DESIGNS],
                    value=list(EXPERIMENT_DESIGNS.keys())[0],
                    clearable=False
                ),
                html.H6("Machine learning experiment"),
                dcc.Dropdown(id="ml_dropdown",
                    options=[{"label": i, "value": i} for i in LEARN_CONFIG["Algos"]],
                    value=list(LEARN_CONFIG["Algos"].keys())[0],
                    clearable=False
                ),
                html.H6("Experiment number"),
                dcc.Dropdown(id="experiment_dropdown",
                    options=[{"label": i, "value": i} for i in range(N_SPLITS)],
                    value="0",
                    clearable=False
                ),
                #html.H6("Show QCs"),
                #dcc.Checklist(id="show_qc_checklist",
                #    options=[
                #        {"label": "QC all run", "value": "true"} # Not sure how to label these 
                #    ]
                #),
                html.H6("Current view info", style={"marginTop": 25}),
                html.Div(id="view_info", children=""),
                html.H6("Global metrics", style={"marginTop": 25}),
                html.Button("Compute", id="compute_global_metrics"),
                dcc.Loading(
                    id="loading_global_metrics",
                    children=html.Div(id="global_metrics", children=""),
                    type="circle"
                )
            ]),
            html.Div(id="main_plots-content", className="six columns", children=[
                dcc.Tabs([
                    dcc.Tab(
                        label="Global view",
                        children=[
                            dcc.Loading(
                                dcc.Graph(
                                    id="accuracy_overview"
                                ),
                                type="circle"
                            ),
                            dcc.Loading(
                                dash_table.DataTable(
                                    id="overview_table",
                                    columns=[{"name": i, "id": i} for i in ["Feature", "Number of models"]],
                                    style_table={
                                        'maxHeight': '300px',
                                        'overflowY': 'scroll'
                                    },
                                    style_as_list_view=True,
                                    style_cell={
                                        'padding': '5px', 
                                        "textAlign": "center"
                                    },
                                )
                            )
                        ]
                    ),
                    dcc.Tab(
                        label="Individual view",
                        children=[
                            # PCA
                            html.H5(id="PCA_title", children="PCA_TITLE"), # Should we put the title on the plot?
                            dcc.Loading(
                                dcc.Graph(id="PCA",
                                    figure=go.Figure(
                                        data=[
                                            go.Scatter(
                                                x=[0,1,3,4],
                                                y=[0,2,3,4]
                                            )
                                        ]
                                    )
                                )
                            ),
                            #Heat-map.
                            html.H5(id="heatmap_title", children="HEAT-MAP TITLE"), # Should we put the title on the plot?
                            dcc.Loading(
                                html.Img(id='heatmap')
                                #dcc.Graph(id="heatmap",
                                #    figure=go.Figure(
                                #        data=[
                                #            go.Scatter(
                                #                x=[0,1,3,4],
                                #                y=[0,2,3,4]
                                #            )
                                #        ]
                                #    )
                                #)
                            ) 
                        ]
                    )
                ]),
                
            ]),
            html.Div(id="side-content", className="four columns", children=[
                html.H6("Experiment metrics"),
                html.Div(id="metrics_table", 
                    style={"margin": "auto", "display": "flex", "justify-content": "center"}
                ),
                html.H6("Metabolite level"),
                html.Div(id="metabolite_dropdown_container", children=[
                    dcc.Dropdown(id="metabolite_dropdown",
                        options=[],
                        clearable=False
                    )
                ]),
                dcc.Graph(id="metabo_boxplot",
                    figure=go.Figure(
                        data=[
                            go.Box(x=[1,2,3,4,5,5,5,5,5,5,5,6,7,8]),
                            go.Box(x=[10,10,10,10,9,9,9,9,9,7,8,5,12,12,15])
                        ]
                    )
                )
            ])
        ]),
    ])

app.layout = layout()

@app.callback(
    Output("metrics_table", "children"),
    [Input("design_dropdown", "value"),
    Input("ml_dropdown", "value"),
    Input("experiment_dropdown", "value")]
)
def get_experiment_statistics(exp_design, ml_exp_name, ml_exp_number):
    # TODO: Increase margins bottom.
    pos_label = EXPERIMENT_DESIGNS[exp_design]["positive_class"]

    model_filename = os.path.join("Results", 
        "{}_{}_{}.pkl".format(exp_design, ml_exp_number, ml_exp_name))

    data_matrix_filename = os.path.join("Splits",
        "{}_{}".format(exp_design, ml_exp_number))
    
    # Load GridSearch
    with open(model_filename, "rb") as fi:
        gc = pkl.load(fi)
        #train_predict = pkl.load(fi)
        #test_predict = pkl.load(fi)

    # Load data matrix
    with open(data_matrix_filename, "rb") as fi:
        X_train = pkl.load(fi)
        y_train = pkl.load(fi)
        X_test = pkl.load(fi)
        y_test = pkl.load(fi)

    table = []
    table_style = {"padding": "12px 55px", "text-align": "left"}
    table.append(html.Tr([html.Th("Metric", style=table_style), html.Th("Train"), html.Th("Test")]))
    if isinstance(gc.best_estimator_, LinearSVC): # No predict_proba for LinearSVC so we use the distance to the margin.
        y_margins = gc.best_estimator_.decision_function(X_test)
        y_test_pred_prob = (y_margins - y_margins.min()) / (y_margins.max() - y_margins.min())
        y_margins = gc.best_estimator_.decision_function(X_train)
        y_train_pred_prob = (y_margins - y_margins.min()) / (y_margins.max() - y_margins.min())
    else:
        y_train_pred_prob = [i[1] for i in gc.predict_proba(X_train)]
        y_test_pred_prob = [i[1] for i in gc.predict_proba(X_test)]
    
    y_train_pred = gc.predict(X_train)
    y_test_pred =  gc.predict(X_test)
    for stat in STATISTICS:
        if STATISTICS[stat] == roc_auc_score:
            #print("y_train shape: {}".format(y_train.shape))
            #print("y_train_pred_prob shape: {}".format(len(y_train_pred_prob)))
            metric_value_train = STATISTICS[stat](y_true=y_train, y_score=y_train_pred_prob)
            metric_value_test = STATISTICS[stat](y_true = y_test, y_score= y_test_pred_prob)
        elif STATISTICS[stat] in [recall_score, f1_score, precision_score]:
            metric_value_train = STATISTICS[stat](y_train, y_train_pred, pos_label=pos_label )
            metric_value_test = STATISTICS[stat](y_test, y_test_pred, pos_label=pos_label)
        else:
            metric_value_train = STATISTICS[stat](y_train, y_train_pred)
            metric_value_test = STATISTICS[stat](y_test, y_test_pred)
        
        if isinstance(metric_value_test, float):
            metric_value_test = "{:0.2f}".format(metric_value_test)
        if isinstance(metric_value_train, float):
            metric_value_train = "{:0.2f}".format(metric_value_train)
        table.append(html.Tr([html.Td(stat), 
            html.Td(metric_value_train), 
            html.Td(metric_value_test)]
        ))
    return html.Table(table)

# def get_experiment_statistics(y_true_train, y_pred_train, y_true_test, y_pred_test):
#     # TODO: Increase margins bottom.
#     table = []
#     table_style = {"padding": "12px 55px", "text-align": "left"}
#     table.append(html.Tr([html.Th("Metric", style=table_style), html.Th("Train"), html.Th("Test")]))
#     for stat in STATISTICS:
#         if STATISTICS[stat] == roc_auc_score:
#             metric_value_train = STATISTICS[stat](y_true_train, y_pred_train)
#             metric_value_test = STATISTICS[stat](y_true_test, y_pred_test)
#         else:
#             metric_value_train = STATISTICS[stat](y_true_train, y_pred_train)
#             metric_value_test = STATISTICS[stat](y_true_test, y_pred_test)
        
#         if isinstance(metric_value_test, float):
#             metric_value_test = "{:0.2f}".format(metric_value_test)
#         if isinstance(metric_value_train, float):
#             metric_value_train = "{:0.2f}".format(metric_value_train)
#         #new_div = html.Div("{}: {}, {}".format(stat, metric_value_train, metric_value_test))
#         table.append(html.Tr([html.Td(stat), 
#             html.Td(metric_value_train), 
#             html.Td(metric_value_test)]
#         ))
#     return html.Table(table)
#     #return children

@app.callback(
    [Output("accuracy_overview", "figure"),
    Output("overview_table", "data")],
    [Input("ml_dropdown", "value"),
    Input("design_dropdown", "value")]
)
def show_global_view(ml_dropdown, design_dropdown):
    print("Updating global accuracy plot")
    splits_name = []
    split_train_accuracy = []
    split_test_accuracy = []

    features = []
    data_matrix_filename = os.path.join("Splits", design_dropdown + "_0")
    with open(data_matrix_filename, "rb") as fi:
        train_df = pkl.load(fi)
        train_targets = pkl.load(fi)
        test_df = pkl.load(fi)
        test_targets = pkl.load(fi)

    reducer = umap.UMAP()

    def filter_cluster(df, threshold=0.5):
        """
        threshold : (proportion) minimum of non-zero values in a line to consider keeping this line
        for example -> threshold = 0.6 means it will keep only the lines where there is at least 60% of non-zero values
        """
        df = df.T
        nbr_col = len(df.columns.to_list())
        df_filtered = df.loc[df.astype(bool).sum(axis=1) >= nbr_col*threshold]
        return df_filtered.T

    train_df = filter_cluster(train_df, threshold=1.0)
    embedding = reducer.fit_transform(train_df)

    trace_train = go.Scatter(
        x=embedding[:, 0],
        y=embedding[:, 1], 
        mode="markers",
        text=train_df.index
    )
    fig = go.Figure(data=[trace_train])
    #return(fig, dash.no_update)

    ###########################
    for model_filename in glob.glob(os.path.join("Results", design_dropdown+"_*_"+ml_dropdown+"*")):
        with open(model_filename, "rb") as fi:
            gc = pkl.load(fi)
            print(gc.best_estimator_.classes_)
            train_predict = pkl.load(fi)
            test_predict = pkl.load(fi)     

        data_matrix_filename = os.path.join("Splits", design_dropdown + "_" + model_filename.split("_")[1])
        with open(data_matrix_filename, "rb") as fi:
            train_df = pkl.load(fi)
            train_targets = pkl.load(fi)
            test_df = pkl.load(fi)
            test_targets = pkl.load(fi)

        splits_name.append(model_filename.split("_")[1])
        split_train_accuracy.append(accuracy_score(y_true=train_targets, y_pred = train_predict))
        split_test_accuracy.append(accuracy_score(y_true=test_targets, y_pred=test_predict))

        if isinstance(gc.best_estimator_, RandomForestClassifier) or \
            isinstance(gc.best_estimator_, DecisionTreeClassifier):
            features_importance = gc.best_estimator_.feature_importances_
        
        zipped = zip(features_importance, train_df.columns)
        zipped = sorted(zipped, key = lambda t:t[0])

        [features.append(i[1]) for i in zipped if np.abs(i[0]) > 0.0]
        
    features_count = Counter(features)
    
    #table = []
    #table_style = {"padding": "12px 55px", "text-align": "left"}
    #table.append(html.Tr([html.Th("Feature", style=table_style), html.Th("Number of models")]))
    #for f in features_count.most_common():
    #    table.append(html.Tr([html.Td(f[0]), html.Td(f[1])]))
    features_column = []
    n_models_column = []
    for f in features_count.most_common():
        features_column.append(f[0])
        n_models_column.append(f[1])

    df = pd.DataFrame()
    df["Feature"] = features_column
    df["Number of models"] = n_models_column

    trace_train = go.Scatter(
        y=split_train_accuracy,
        name="Train accuracy"
    )
    trace_test = go.Scatter(
        y=split_test_accuracy,
        name="Test accuracy"
    )
    #fig = go.Figure(data=[trace_train, trace_test]) #Uncomment for accuracy plot instead of UMAP. TODO: add choice (maybe config file?
    return fig, df.to_dict("records")


@app.callback(
    Output("global_metrics", "children"),
    [Input("compute_global_metrics", "n_clicks"),
    Input("ml_dropdown", "value"),
    Input("design_dropdown", "value")]
)
def compute_global_metrics(n_clicks, ml_algo, exp_design):
    pos_label = EXPERIMENT_DESIGNS[exp_design]["positive_class"]
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    if dash.callback_context.triggered[0]['prop_id'].split('.')[0] != "compute_global_metrics":
        return ""
    data_matrix_file_list = glob.glob(os.path.join("Splits", "{}_*".format(exp_design)))
    metrics_results_train =  {i:[] for i in STATISTICS}
    metrics_results_test =  {i:[] for i in STATISTICS}

    for data_matrix_filename in data_matrix_file_list:
        split_number = data_matrix_filename.split("_")[-1]
        model_filename = os.path.join("Results", 
            "{}_{}_{}.pkl".format(exp_design, split_number, ml_algo))
    
        with open(data_matrix_filename, "rb") as fi:
            X_train = pkl.load(fi)
            y_train = pkl.load(fi)
            X_test = pkl.load(fi)
            y_test = pkl.load(fi)

        with open(model_filename, "rb") as fi:
            gc = pkl.load(fi)
            #print(gc.best_estimator_.classes_)
            y_train_pred = pkl.load(fi)
            y_test_pred = pkl.load(fi)

        if isinstance(gc.best_estimator_, LinearSVC):
            y_margins = gc.best_estimator_.decision_function(X_test)
            y_test_pred_prob = (y_margins - y_margins.min()) / (y_margins.max() - y_margins.min())
            y_margins = gc.best_estimator_.decision_function(X_train)
            y_train_pred_prob = (y_margins - y_margins.min()) / (y_margins.max() - y_margins.min())
        else:
            y_train_pred_prob = [i[1] for i in gc.predict_proba(X_train)]
            y_test_pred_prob = [i[1] for i in gc.predict_proba(X_test)]

        for stat in STATISTICS:
            if STATISTICS[stat] == roc_auc_score:
                #print("y_train shape: {}".format(y_train.shape))
                #print("y_train_pred_prob shape: {}".format(len(y_train_pred_prob)))
                metric_value_train = STATISTICS[stat](y_true=y_train, y_score=y_train_pred_prob)
                metric_value_test = STATISTICS[stat](y_true = y_test, y_score= y_test_pred_prob)
            elif STATISTICS[stat] in [recall_score, f1_score, precision_score]:
                metric_value_train = STATISTICS[stat](y_train, y_train_pred, pos_label=pos_label )
                metric_value_test = STATISTICS[stat](y_test, y_test_pred, pos_label=pos_label)
            else:
                metric_value_train = STATISTICS[stat](y_train, y_train_pred)
                metric_value_test = STATISTICS[stat](y_test, y_test_pred)

            metrics_results_train[stat].append(metric_value_train)
            metrics_results_test[stat].append(metric_value_test)

    table = []
    table_style = {"padding": "12px 55px", "text-align": "left"}
    table.append(html.Tr([html.Th("Metric", style=table_style), html.Th("Train"), html.Th("Test")]))
    for stat in STATISTICS:
        #(stat)
        train_average = np.average(metrics_results_train[stat])
        test_average = np.average(metrics_results_test[stat])
        train_std = np.std(metrics_results_train[stat])
        test_std = np.std(metrics_results_test[stat])
        #print(train_average, train_std)
        table.append(html.Tr([html.Td(stat), 
            html.Td("{:0.2f} ({:0.2f})".format(train_average, train_std)), 
            html.Td("{:0.2f} ({:0.2f})".format(test_average, test_std))]
        ))
    return html.Table(table)




@app.callback(
    Output("metabo_boxplot", "figure"),
    [Input("metabolite_dropdown", "value")],
    [State("design_dropdown", "value"),
    State("experiment_dropdown", "value")]
)
def update_boxplot_metabolite(metabolite_name, exp_design, ml_exp_number):
    data_matrix_filename = os.path.join("Splits",
        "{}_{}".format(exp_design, ml_exp_number))
    
    # Load data matrix
    with open(data_matrix_filename, "rb") as fi:
        train_df = pkl.load(fi)
        train_targets = pkl.load(fi)
        test_df = pkl.load(fi)
        test_targets = pkl.load(fi)

    merged_df = train_df.append([test_df])
    merged_targets = train_targets + test_targets
    if metabolite_name is None:
        return dash.no_update
    selected_metabo_data = np.array(merged_df[metabolite_name])

    box_traces = []
    for i in set(merged_targets):
        box_traces.append(go.Box(
            x=selected_metabo_data[np.array(merged_targets) == i],
            name=i)
        )
    layout = go.Layout(title="Abunce level of the selected metabolite between classes")
    return go.Figure(data=box_traces, layout=layout)



@app.callback(
    [Output("PCA", "figure"),
    Output("PCA_title", "children"),
    Output("metabolite_dropdown", "options"),
    Output("metabolite_dropdown", "value"),
    Output("heatmap", "src"),
    Output("heatmap_title", "children"),
    #Output("metrics_table", "children"),
    Output("view_info", "children")],
    [Input("design_dropdown", "value"),
    Input("ml_dropdown", "value"),
    Input("experiment_dropdown", "value")]
    #Input("show_qc_checklist", "value")]
)
def update_core(exp_design, ml_exp_name, ml_exp_number):
    model_filename = os.path.join("Results", 
        "{}_{}_{}.pkl".format(exp_design, ml_exp_number, ml_exp_name))

    data_matrix_filename = os.path.join("Splits",
        "{}_{}".format(exp_design, ml_exp_number))
    
    # Load GridSearch
    with open(model_filename, "rb") as fi:
        gc = pkl.load(fi)
        train_predict = pkl.load(fi)
        test_predict = pkl.load(fi)

    # Load data matrix
    with open(data_matrix_filename, "rb") as fi:
        train_df = pkl.load(fi)
        train_targets = pkl.load(fi)
        test_df = pkl.load(fi)
        test_targets = pkl.load(fi)

    # Prepare View Info

    view_info_children = []
    view_info_children.append(html.Div("{} samples in the dataset".format(len(train_targets) + len(test_targets))))
    view_info_children.append(html.Div("{} samples in the training set".format(len(train_targets))))
    view_info_children.append(html.Div("{} samples in the testing set".format( len(test_targets))))


    # Compute Metrics table
    #metric_table = get_experiment_statistics(train_targets, train_predict, \
    #    test_targets, test_predict)

    # Filter important features
    if isinstance(gc.best_estimator_, RandomForestClassifier) or \
            isinstance(gc.best_estimator_, DecisionTreeClassifier):
        features_importance = gc.best_estimator_.feature_importances_
    else:
        raise ValueError("Check code and define features extraction process.")

    zipped = zip(features_importance, train_df.columns)
    zipped = sorted(zipped, key = lambda t:t[0])

    if np.sum(features_importance > 0.0) < NUMBER_FEATURE_TO_KEEP_FOR_PCA:
        important_index = [i[1] for i in zipped[-1 * np.sum(features_importance > 0.0):]]
        
    else:
        important_index = [i[1] for i in zipped[-1 * NUMBER_FEATURE_TO_KEEP_FOR_PCA:]]
    
    train_df_filtered = train_df[important_index]
    test_df_filtered = test_df[important_index]

    # Compute PCA
    merged_df = train_df_filtered.append([test_df_filtered])
    
    std_clf = make_pipeline(StandardScaler(), PCA(n_components=2))
    std_clf.fit(train_df_filtered)
    train_pca_transformed = std_clf.transform(train_df_filtered)
    test_pca_transformed = std_clf.transform(test_df_filtered)

    # Create PCA plot
    merged_targets = np.array(train_targets + test_targets)
    train_targets = np.array(train_targets)
    test_targets = np.array(test_targets)
    
    pca_traces = []
    for c in gc.classes_:
        trace = go.Scatter(
            x=train_pca_transformed[train_targets==c, 0],
            y=train_pca_transformed[train_targets==c, 1],
            name="{} training".format(c),
            mode="markers"
        )
        pca_traces.append(trace)
        trace = go.Scatter(
            x=test_pca_transformed[test_targets==c, 0],
            y=test_pca_transformed[test_targets==c, 1],
            name="{} testing".format(c),
            mode="markers"
        )
        pca_traces.append(trace)

    figure_pca = go.Figure(data=pca_traces)
    pca_title = "PCA plot using {} metabolites selected using the {} algorithm on split {}".format(
        train_df_filtered.shape[1], ml_exp_name, ml_exp_number)

    # Create Metabolite dropdown
    metabolite_dropdown = [{"label": i, "value": i} for i in np.concatenate([important_index, train_df.columns])]
    selected_metabolite_value = important_index[0]

    # Create heatmap using merged DF
    heatmap = get_static_heatmap_plot(StandardScaler().fit_transform(merged_df), \
        merged_df.columns.values, merged_df.index, merged_targets)
    heatmap_title = "Heat-map of {} metabolite normalized abundnce across the different samples".format(
        train_df_filtered.shape[1])
    #Return values
    return figure_pca, pca_title, metabolite_dropdown, selected_metabolite_value, \
        heatmap, heatmap_title, view_info_children


def get_static_heatmap_plot(NpArray, Label_X, Label_Y, targets):
    target_colors = dict(zip(set(targets), sns.color_palette("Set2")))
    row_colors = [target_colors[i] for i in targets]
    heatmap = sns.clustermap(pd.DataFrame(NpArray, columns=Label_X, index=Label_Y),
         row_colors=row_colors, cmap="RdBu_r")
    fig = heatmap.fig
    buf = io.BytesIO() # in-memory files
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    return "data:image/png;base64,{}".format(data)

def get_heatmap_plot(NpArray, Label_X, Label_Y):
    component = dash_bio.Clustergram(
        hidden_labels=['row'],
        color_threshold={'row': 150, 'col': 700},
        data=NpArray,
        column_labels=list(Label_X),
        width=700,
        height=800,
        row_labels=list(Label_Y),
        optimal_leaf_order =True
    )
    return component

if __name__ == "__main__":
    app.run_server(debug=True)
