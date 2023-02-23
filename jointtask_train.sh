python run_model.py -mode train -model_checkpoint allenai/tk-instruct-base-def-pos \
-experiment_name check_dummy -task joint -output_dir ./Models \
-id_tr_data_path /scratch/sgoyal41/InstructABSA/Dataset/test_set.csv \
-id_te_data_path /scratch/sgoyal41/InstructABSA/Dataset/test_set.csv \
-per_device_train_batch_size 16 -per_device_eval_batch_size 16 -num_train_epochs 4