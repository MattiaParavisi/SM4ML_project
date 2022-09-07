# Statistical Methods 4 Machine Learning Project
## _Mattia Paravisi, matr: 08395A_
### Readme file

How to setup the project enviroment, maybe different from the one i used:
- Unzip the file CatsDogs.zip (https://unimibox.unimi.it/index.php/s/eNGYGSYmqynNMqF) at the same level of scripts folder
- Run scripts/data\_analysis\_scripts/data\_analysis.py with $ python3 data\_analysis.py.
- A new folder called CatsDogs_resized will be created on top level
- Use the following command in bash $ splitfolders --output 'NN\_data' --ratio .8 .1 .1 -- CatsDogs\_resized/
- Now you can use all the scripts in scripts folder

Faster way to setup the project enviroment, same i used:
- Unzip the file CatsDogs.zip (https://unimibox.unimi.it/index.php/s/eNGYGSYmqynNMqF) at the same level of scripts folder
- Download both files from https://unimi2013-my.sharepoint.com/:f:/g/personal/mattia_paravisi_studenti_unimi_it/EtMcuaElM85CtcDbpttHFdkBkf_IPsNrHgJ-burTIi1UdQ?e=TzIVwg
- Unzip at same level of scripts folder
- Run scripts/data\_analysis\_scripts/data\_analysis\_if\_download.py with $ python3 data\_analysis\_if\_download.py.

In both cases folder structure must be:

<pre>
.
├── notebook.ipynb
├── README.md
├── scripts
│   ├── cnn_scripts
│   │   ├── batch analysis
│   │   │   └── batch_analysis.py
│   │   ├── best_conv_filters_number_analysis
│   │   │   └── best_conv_filters_number.py
│   │   ├── best_dense_layer_analysis
│   │   │   └── best_dense_layer_analysis.py
│   │   ├── best_model_analysis
│   │   │   └── best_model.py
│   │   ├── dropout_layer_analysis
│   │   │   └── dropout_layer_analysis.py
│   │   ├── epochs_analysis
│   │   │   └── epochs_analysis.py
│   │   ├── filters_number_analysis
│   │   │   └── filters_number_analysis.py
│   │   ├── incr_decr_conv2d_filters_analysis
│   │   │   └── incr_decr_conv2d_filters_analysis.py
│   │   ├── kfold
│   │   │   └── kfold.py
│   │   └── overfitted_model
│   │       └── overfitted_model.py
│   └── data_analysis_scripts
│       ├── data_analysis_if_download.py
│       └── data_analysis.py
├── SM4ML_project.pdf
└── SM4ML project.zip

</pre>
