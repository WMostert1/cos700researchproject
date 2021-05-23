import openml
import csv
from shutil import copyfile
MAX_INSTANCES = 20000
with open('./openml_datasets/dataset_metadata.csv', mode='w') as dataset_metadata_file:
    meta_writer = csv.writer(dataset_metadata_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    metadata_labels = ['did','name', 'NumberOfClasses', 'NumberOfFeatures', 'NumberOfInstances', 'NumberOfInstancesWithMissingValues',
                       'NumberOfMissingValues', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures']
    meta_writer.writerow(metadata_labels)
    datasets_dict = openml.datasets.list_datasets(output_format="dict")
    index = 0
    for ds_id in datasets_dict:
        meta = datasets_dict.get(ds_id)
        if meta.get('format', "") != 'ARFF':
            print(f"Skipping datasets that does not have an ARFF format: {meta['name']}")
            index += 1
            continue

        if meta.get('NumberOfInstances', MAX_INSTANCES + 1) > MAX_INSTANCES:
            print(f"Skipping dataset with more than {MAX_INSTANCES} instances: {meta['name']}")
            index += 1
            continue

        if meta.get('NumberOfClasses', 0) == 0:
            print(f"Skipping dataset that has no classes: {meta['name']}")
            index += 1
            continue

        dataset = openml.datasets.get_dataset(dataset_id=ds_id)
        copyfile(dataset.data_file, f"./openml_datasets/{dataset.name}-{ds_id}.arff")
        vals = []
        for label in metadata_labels:
            vals.append(meta.get(label, ""))
        meta_writer.writerow(vals)
        index += 1
        print(f"{index}/{len(datasets_dict)}")