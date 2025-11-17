import os
from models.a3tgcn_revised import A3TGCNCat1
from utils.processing_utils import fully_connected_edge_index_batched, mi_edge_index_batched
from utils.device_set import device_set

if __name__ == "__main__":
    device = device_set()
    CURDIR = os.path.dirname(__file__)
    
    from teds_tensor_dataset import TEDSTensorDataset
    from train_eval_a3tgcn_revised import train_test_split_customed

    root = os.path.join(CURDIR, 'data_tensor_cache')
    dataset = TEDSTensorDataset(root)

    train_dataloader, val_dataloader, test_dataloader = train_test_split_customed(dataset, batch_size=32)
    
    col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info

    BATCH_SIZE = 32

    model = A3TGCNCat1(batch_size=BATCH_SIZE, col_list=col_list,
                       col_dims=col_dims, embedding_dim=64, hidden_channel=64)
    model.to(device)

    counter = 0

    # template_edge_index = fully_connected_edge_index_batched(num_nodes=60, batch_size=BATCH_SIZE)
    # template_edge_index = template_edge_index.to(device)

    mi_dict_path = os.path.join(CURDIR, 'data', 'mi_dict_static.pickle')
    mi_edge_index = mi_edge_index_batched(batch_size=BATCH_SIZE, 
                                          mi_dict_path=mi_dict_path, 
                                          top_k=6, 
                                          return_edge_attr=False)
    mi_edge_index = mi_edge_index.to(device)

    for x_batch, y_batch, los_batch in train_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        los_batch = los_batch.to(device)

        if counter == 3: break
        result = model(
            ad_col_index,
            dis_col_index,
            x_batch,
            los_batch,
            mi_edge_index,
            device
        )
        print(result)
        counter += 1