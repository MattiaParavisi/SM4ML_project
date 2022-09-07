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
SM4ML_project
--------->|
--------->|notebook.ipynb
--------->|README.md
--------->|SM4ML project.zip
--------->|SM4ML_project.pdf
--------->|CatsDogs
--------->--------->|Cats
--------->--------->|Dogs
--------->|CatsDogs_resized
--------->--------->|Cats_resized
--------->--------->|Dogs_resized
--------->|NN_data
--------->--------->|train
--------->--------->|test
--------->--------->|val
--------->|scripts
--------->--------->|cnn_scripts
--------->--------->|data\_analysis\_scripts
</pre>
