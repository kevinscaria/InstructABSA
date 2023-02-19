python train.py -model_checkpoint allenai/tk-instruct-base-def-pos -experiment_name laptops_instruct_temp \
-task ate -output_dir ./Models \
-id_tr_data_path ./semeval14/ABSA_TrainData/Restaurants_Train_v2.csv \
-id_te_data_path ./semeval14/ABSA_Gold_TestData/Restaurants_Test_Gold.csv \
-ood_tr_data_path ./semeval14/ABSA_TrainData/Laptop_Train_v2.csv \
-ood_te_data_path ./semeval14/ABSA_Gold_TestData/Laptops_Test_Gold.csv \
-per_device_train_batch_size 16 -per_device_eval_batch_size 16 -num_train_epochs 4