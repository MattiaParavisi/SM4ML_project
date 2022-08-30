# SM4ML_project
Repository containing project files for the course "statistical methods for machine learning"

How to run the project:
-> Unzip the file CatsDogs.zip at the same level of scripts folder, the hierarchy will be:
SM4ML_project
|__scripts
|__CatsDogs
-> Run the script in scripts/data_analysis.py. A new folder called CatsDogs_resized will be created on top level
SM4ML_project
|__scripts
|__CatsDogs
|__CatsDogs_resized
-> Use the following command in bash "splitfolders --output 'NN_data' --ratio .8 .1 .1 -- CatsDogs_resized/"
SM4ML_project
|__scripts
|__CatsDogs
|__CatsDogs_resized
|__NN_data
-> Now you can use all the scripts in scripts folder
