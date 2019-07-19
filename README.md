# Machine Learning Model Search

Codebase for running a parameter search over multiple machine learning models, using either classification or regression.

Example usage:

- To run a grid search on regression models, edit "src/models/config_regression.yaml" to set the hyperparameters and models you want to search over. Also edit the metrics you want to save out, at the bottom of the yaml file.
- The grid search can then be run using grid_search_regression.ipynb.
- Results of the search will be saved in src/results/ in a folder named using the current date and time (this ensures it won't be over-written on subsequent runs).
- It is suggested that you add the results folder to your gitignore.
- To visualise the results, run "viz_results_regression.ipynb" in src/visualisations
