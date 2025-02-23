SHELL := /bin/bash


data/raw_datasets/${NUM}:
	@if [[ $${NUM: -1} == "M" ]]; then \
		python3 scripts/data_creation/dataset_creation.py --number_of_equations $${NUM:0:$${#NUM}-1}000000 --no-debug; \
	elif [[ $${NUM: -1} == "K" ]]; then \
		python3 scripts/data_creation/dataset_creation.py --number_of_equations $${NUM:0:$${#NUM}-1}000 --no-debug; \
	else echo "Error only M or K is allowed"; \
	fi
		

# data/datasets/${NUM}/.dirstamp: data/raw_datasets/${NUM}
# 	python3 scripts/data_creation/split_train_val.py --data_path $?

data/datasets/${NUM}/${NUM}_subset: data/datasets/${NUM}/.dirstamp
	python3 scripts/data_creation/filter_validation.py --val_path data/datasets/${NUM}/${NUM}_val

#This is used for filtering a small number from an already existing dataset
data/datasets/${NUM}/${NUM}_filtered: data/raw_datasets/${NUM}
	python3 scripts/data_creation/filter_from_already_existing.py --data_path data/raw_datasets/${NUM}

data/datasets/${NUM}/.dirstamp_hdf: data/datasets/${NUM}/${NUM}_subset
	python3 scripts/data_creation/to_h5.py --folder_dataset data/datasets/${NUM}

data/benchmarks/.dirstamp: 
	python3 scripts/data_creation/filter_from_already_existing.py --data_path data/raw_datasets/${NUM}

