# LipidTrends
Trend plot for lipids

This is to plot the trend plot with clustering of lipids based on the corresponding abundances.
Supports the following clustering algorithms:

- `hew` (Hierarchical clustering with Ward linkage)
- `km` (KMeans)
- `bisect_km` (Bisecting KMeans)
- `gmm` (Gaussian Mixture Model)
- `dpgmm` (Dirichlet Process Gaussian Mixture Model)

*Above clustering algorithms are implemented using `scikit-learn` library. and can be defined in the `LipidTrends.ipynb` notebook file.*

**Important note:**

Due to the principles of the `km`, `bisect_km`, `gmm`, and `dpgmm` clustering algorithms, the number of clusters `k` should be specified prior to the analysis.

We provide the possibility to generate a series of plots with different `k` values to help the user to decide on the optimal number of clusters.

*Many of the clustering algorithms require a random state to start the clustering.*

*Due to the version of the libraries and the local computer system setups, the results may vary slightly between different runs.*

We provide the possibility to set the random state to enhance reproducibility of the results. 
However, there is still possibility to obtain slightly differed cluster results.

*Please keep this in mind when interpreting the results and we recommend using the same device and environment for the whole analysis.*

+ Please keep using the same device and environment for the whole analysis.

+ Please keep all the parameters and the exported in between files to keep a record of the analysis.

+ You can also copy the whole folder for each new dataset to keep the record of the analysis managed by task/project.


#### Requirements
**Recommended: Use the `requirements.txt` file**

```bash
pip install -r requirements.txt
```
**Alternative: Install the dependencies manually**

```bash
pip install matplotlib plotly pandas scikit-learn umap-learn natsort jupyterlab jupyter openpyxl kaleido
```
*Code compatibility tested: Python `3.11` and `3.12` using `miniconda`*

#### Usage

+ Run the jupyter notebook `LipidTrends.ipynb`

+ Follow the instructions in the notebook, change the parameters accordingly.


#### Copyright (C) 2021-2024  LMAI_team @ TU Dresden:

+ LMAI_team: Zhixu Ni, Maria Fedorova


#### Licensing:

+ This code is licensed under AGPL-3.0 license (Affero General Public License v3.0).
  - For more information, please read:
  - AGPL-3.0 License: [https://www.gnu.org/licenses/agpl-3.0.en.html](https://www.gnu.org/licenses/agpl-3.0.en.html)


#### Citation:

+ Please cite our publication in an appropriate form.


#### For more information, please contact:

+ Fedorova Lab (#LMAI_team): [https://fedorovalab.net/](https://fedorovalab.net/)
+ LMAI on Github: [https://github.com/LMAI-TUD](https://github.com/LMAI-TUD)