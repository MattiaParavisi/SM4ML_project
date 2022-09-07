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
├── CatsDogs
│   ├── Cats
│   └── Dogs
├── CatsDogs_resized
│   ├── Cats_resized
│   └── Dogs_resized
├── NN_data
│   ├── test
│   │   ├── Cats_resized
│   │   └── Dogs_resized
│   ├── train
│   │   ├── Cats_resized
│   │   └── Dogs_resized
│   └── val
│       ├── Cats_resized
│       └── Dogs_resized
├── notebook.ipynb
├── README.md
├── scripts
│   ├── cnn_scripts
│   │   ├── batch analysis
│   │   ├── best_conv_filters_number_analysis
│   │   ├── best_dense_layer_analysis
│   │   ├── best_model_analysis
│   │   ├── dropout_layer_analysis
│   │   ├── epochs_analysis
│   │   ├── filters_number_analysis
│   │   ├── incr_decr_conv2d_filters_analysis
│   │   ├── kfold
│   │   └── overfitted_model
│   └── data_analysis_scripts
├── SM4ML_project.pdf
└── SM4ML project.zip
</pre>
