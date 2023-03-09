python ../run_model.py -mode train -model_checkpoint allenai/tk-instruct-base-def-pos \
-experiment_name atsc_check -task atsc -output_dir ../Models \
-inst_type 2 \
-id_tr_data_path ../Dataset/example_data.csv \
-id_te_data_path ../Dataset/example_data.csv \
-per_device_train_batch_size 16 -per_device_eval_batch_size 16 -num_train_epochs 4