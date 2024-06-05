import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import Dataset, InMemoryDataset
import torch.utils.data as data
import os
import pickle
import torch_geometric
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm 
from torch_geometric.transforms import Polar, Distance

class GraphDataset(Dataset):
    def __init__(self, root, raw_folder_name, processed_folder_name, n_c, 
                y_name='condition', condition_mapping=None, test_size=0.35,
                transform=None, pre_transform=None):

        self.root = root
        self.raw_folder_name = raw_folder_name
        self.processed_folder_name = processed_folder_name
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.test_size=test_size
        self.condition_mapping = condition_mapping
        try:
            self.num_classes = n_c
        except:pass
        self.y_name = y_name
        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_folder_name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_folder_name)

    @property
    def raw_file_names(self):
        return sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.pkl')])

    @property
    def processed_file_names(self):
        # Only files for full graphs
        return sorted([f for f in os.listdir(self.processed_dir) if f.endswith('.gpt')])

    def len(self):
        return len(self.processed_paths)

    def process(self):
        """Featurize all cellular graphs"""
        for raw_path in tqdm(self.raw_paths):
            with open(raw_path, 'rb') as f:
                G = pickle.load(f)

                # Get info
                info = raw_path.split('\\')[-1].split('_')

                # Creat torch data
                data = torch_geometric.utils.from_networkx(G)
                data.condition = torch.tensor(self.condition_mapping[info[0]])
                data.fov = torch.tensor(int(info[1][-1]))
                data.id = torch.tensor(int(info[2][:-4]))

                # Train and test mask 
                X_train, X_test = train_test_split(pd.Series(list(G.nodes())), 
                                                    test_size=self.test_size, 
                                                    random_state=42)
                n_nodes = G.number_of_nodes()
                train_mask = torch.zeros(n_nodes, dtype=torch.bool)
                test_mask = torch.zeros(n_nodes, dtype=torch.bool)
                train_mask[X_train.index] = True
                test_mask[X_test.index] = True
                data['train_mask'] = train_mask
                data['test_mask'] = test_mask

                # Add Polar Transform
                data.pos = (data.pos).to(dtype=torch.float)
                pos_transform = Polar()
                try:
                    data = pos_transform(data)
                except:pass
                torch.save(data, os.path.join(self.processed_dir, f'{data.condition}_{data.fov}_{data.id}.gpt'))
        return

    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        return data

    def get(self, idx):
        filePath = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(filePath)
        data.x = data['labels']
        data.y = torch.tensor([data[self.y_name]])
        data.edge_weight = data['weight']
        data.name = self.processed_file_names[idx]
        return data

class GraphDatasetMLP(GraphDataset):
    def __init__(self, root, raw_folder_name, processed_folder_name, n_c, 
                y_name='condition', condition_mapping=None, test_size=0.35,
                transform=None, pre_transform=None):
        super().__init__(root, raw_folder_name, processed_folder_name, n_c, 
                y_name=y_name, condition_mapping=condition_mapping, test_size=test_size,
                transform=transform, pre_transform=pre_transform)

    def get(self, idx):
        filePath = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(filePath)
        data_new = torch_geometric.data.Data()
        data_new.x = data['labels'].sum(axis=0, keepdim=True)
        data_new.y =  torch.tensor([data[self.y_name]])
        data_new.name = self.processed_file_names[idx]
        return data_new

class GraphDatasetPixelgen(GraphDataset):
    def __init__(self, root, raw_folder_name, processed_folder_name, n_c, 
                y_name='condition', condition_mapping=None, test_size=0.35,
                transform=None, pre_transform=None):
        super().__init__(root, raw_folder_name, processed_folder_name, n_c, 
                y_name=y_name, condition_mapping=condition_mapping, test_size=test_size,
                transform=transform, pre_transform=pre_transform)

    def process(self):
        """Featurize all cellular graphs"""
        for raw_path in tqdm(self.raw_paths):
            with open(raw_path, 'rb') as f:
                G = pickle.load(f)

                # Get info
                info = raw_path.split('\\')[-1].split('_')

                # Creat torch data
                data = torch_geometric.utils.from_networkx(G)
                data.condition = torch.tensor(self.condition_mapping[info[0]])
                data.id = torch.tensor(int(info[1][8:-4]))

                # Train and test mask 
                X_train, X_test = train_test_split(pd.Series(list(G.nodes())), 
                                                    test_size=self.test_size, 
                                                    random_state=42)
                n_nodes = G.number_of_nodes()
                train_mask = torch.zeros(n_nodes, dtype=torch.bool)
                test_mask = torch.zeros(n_nodes, dtype=torch.bool)
                train_mask[X_train.index] = True
                test_mask[X_test.index] = True
                data['train_mask'] = train_mask
                data['test_mask'] = test_mask

                # Add Polar Transform
                data.pos = (data.pos).to(dtype=torch.float)
                pos_transform = Polar()
                try:
                    data = pos_transform(data)
                except:pass
                torch.save(data, os.path.join(self.processed_dir, f'{data.condition}_{data.id}.gpt'))
        return

class GraphDatasetMLP_sub(GraphDataset):
    def __init__(self, root, raw_folder_name, processed_folder_name, n_c, 
                y_name='condition', condition_mapping=None, test_size=0.35,
                transform=None, pre_transform=None):
        super().__init__(root, raw_folder_name, processed_folder_name, n_c, 
                y_name='condition', condition_mapping=None, test_size=0.35,
                transform=None, pre_transform=None)

    def get(self, idx):
        filePath = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(filePath)
        nuclei = data['nuclei'].bool()
        data_new = torch_geometric.data.Data()
        count_cyto = data['labels'][~nuclei].sum(axis=0, keepdim=True)
        count_nuclei = data['labels'][nuclei].sum(axis=0, keepdim=True)
        data_new.x = torch.concat([count_cyto, count_nuclei], axis=1)
        data_new.y = data[self.y_name]
        data_new.name = self.processed_file_names[idx]
        return data_new
    
class GraphDataset2D3D(GraphDataset):
    def __init__(self, root, raw_folder_name, processed_folder_name, n_c, 
                y_name='condition', condition_mapping=None, test_size=0.35,
                transform=None, pre_transform=None):
        super().__init__(root, raw_folder_name, processed_folder_name, n_c, 
                y_name=y_name, condition_mapping=condition_mapping, test_size=test_size,
                transform=transform, pre_transform=pre_transform)

    def process(self):
        """Featurize all cellular graphs"""
        for raw_path in tqdm(self.raw_paths):
            with open(raw_path, 'rb') as f:
                G = pickle.load(f)

                # Get info
                info = raw_path.split('\\')[-1].split('_')
                # Creat torch data
                data = torch_geometric.utils.from_networkx(G)
                data.condition = torch.tensor(self.condition_mapping[info[0]])
                data.fov = torch.tensor(int(info[1][-1]))
                data.id = torch.tensor(int(info[2][:-4]))

                # Add Polar Transform
                data.pos3D = (data.pos).to(dtype=torch.float)
                data.pos = (data.pos[:,:2]).to(dtype=torch.float)
                pos_transform = Distance()
                try:
                    data = pos_transform(data)
                except:pass
                torch.save(data, os.path.join(self.processed_dir, f'{data.condition}_{data.fov}_{data.id}.gpt'))
        return


    def get(self, idx):
        filePath = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(filePath)
        data.x = data['labels']
        data.y = torch.tensor([data[self.y_name]])
        data.edge_weight_2D = data.edge_attr.squeeze()
        data.edge_weight_3D = data['weight']
        data.name = self.processed_file_names[idx]
        return data

class GraphDataset2Dprojected(GraphDataset2D3D):
    def __init__(self, root, raw_folder_name, processed_folder_name, n_c, 
                y_name='condition', condition_mapping=None, test_size=0.35,
                transform=None, pre_transform=None):
        super().__init__(root, raw_folder_name, processed_folder_name, n_c, 
                y_name=y_name, condition_mapping=condition_mapping, test_size=test_size,
                transform=transform, pre_transform=pre_transform)


    def get(self, idx):
        filePath = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(filePath)
        data.x = data['labels']
        data.y = torch.tensor([data[self.y_name]])
        data.edge_weight = data.edge_attr.squeeze()
        data.name = self.processed_file_names[idx]
        return data


def train_test_val_split(dataset, test_ratio=0.4, val_ratio=0.2):
    seed = torch.Generator().manual_seed(42)

    # dataset = dataset.shuffle()
    test_size = int(len(dataset)*test_ratio)
    train_size = len(dataset) - int(len(dataset)*test_ratio)
    train_set, test_set = data.random_split(dataset, [train_size, test_size], generator=seed)

    val_size = int(train_size*val_ratio)
    train_size = train_size - val_size
    train_set, val_set = data.random_split(train_set, [train_size, val_size], generator=seed)

    return train_set, val_set, test_set