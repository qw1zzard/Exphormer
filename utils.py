import dgl
import pandas as pd
import torch
from dgl.data.fraud import FraudAmazonDataset, FraudYelpDataset

from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.loader import DataLoader


DATA_NAMES = ['yelp', 'amazon', 'amazon_new']


class Dataset(InMemoryDataset):
    def __init__(self, data_list, transform=None):
        super(Dataset, self).__init__(None, transform)  # we do not have a root folder
        self.data, self.slices = self.collate(data_list)

    def _download(self):
            pass

    def _process(self):
            pass


def load_data(data_name, raw_dir='./data'):
    assert data_name in DATA_NAMES

    if data_name == 'yelp':
        graph = FraudYelpDataset(raw_dir).graph
    elif data_name.startswith('amazon'):
        graph = FraudAmazonDataset(raw_dir).graph
        if data_name == 'amazon_new':
            features = graph.ndata['feature'].numpy()
            mask_dup = torch.BoolTensor(
                pd.DataFrame(features).duplicated(keep=False).values
            )
            graph = graph.subgraph(~mask_dup)

    edges = graph.edges(etype='net_rur')
    edges = torch.stack((edges[0], edges[1]), dim=0).type(dtype=torch.long)

    pyg_data = Data(
        x=graph.ndata['feature'],
        edge_index=edges,
        y=graph.ndata['label'],
    )


    dataset = Dataset([pyg_data])

    return [
        DataLoader(dataset, batch_size=1, shuffle=True),
        DataLoader(dataset, batch_size=1, shuffle=True),
        DataLoader(dataset, batch_size=1, shuffle=True),
    ]
