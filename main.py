from data.processing_temporal import processing_temporal_main
import os
import pickle

if __name__ == "__main__":
    CURDIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(CURDIR, 'data', 'temporal_graph_data_mi.pickle')
    pickle_dataset = processing_temporal_main()
    with open(DATA_PATH, 'wb') as f:
        pickle.dump(pickle_dataset, f)
        print("pickle_dataset saved!!")
    