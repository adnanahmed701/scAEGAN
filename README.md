

CrossGan-Omics
---------------------------------------------------------------------------------------------------------
This repository contains the CrossGan-Omics code and online data for the single-omics and multi-omics integration. It also contains the code to evaluate and visualize the integration results. Metrics are available for quantifying outputs quality.

* [Summary](#Summary)
* [Datasets](#Datasets)
* [Usage](#Usage)
* [Running Example](#Running-Example)



 Summary
 -------
CrossGan-Omics is a python based deep learning model that is designed for single-cell-omics and multi-omics integration. CrossGan-Omics performs this by using an Autoencoder which learns a low-dimensional embedding of each experiment independently, respecting each sample's uniqueness, protocol. Next, cycleGAN learns a non-linear mapping between these two Autoencoder representations.
