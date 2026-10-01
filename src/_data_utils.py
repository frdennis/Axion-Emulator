from typing import Optional, List, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset, Subset

import pytorch_lightning as pl

#from semu import scalers
import _scalers as scalers



def read_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    label_columns: List[str],
    sample_idx_column: str = None,
    as_tensors: bool = True,
):
    """
    Read a pandas data frame with given feature and label columns

    Args:
        df: pandas dataframe
        feature_columns (List[str]):  list of column names to be used as feature
        label_columns (List[str]): list of column names to be used as labels,
        if one element ends '*' in str, it will select all columns startinwith the string before 
        '*'
        as_tensors (bool): if True, returns a pytorch tensor
    """
    x = df[feature_columns].to_numpy()
    if label_columns[0][-1] == "*":
        label_columns = [
            col for col in df.columns if col.startswith(label_columns[0][:-1])
        ]
    y = df[label_columns].to_numpy()
    if sample_idx_column is not None:
        sample_index = df[sample_idx_column]
    else:
        sample_index = None
    if as_tensors:
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            sample_index,
        )
    return x, y, sample_index


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features: Union[np.array, torch.tensor],
        labels: Union[np.array, torch.tensor],
        sample_index: Union[np.array, torch.tensor] = None,
    ):
        """ Create a pytorch dataset for a set of features and labels,
        making sure these are pytorch tensors

        Args:
            features (Union[np.array, torch.tensor]): features
            labels (Union[np.array, torch.tensor]): labels
        """
        self.x = features
        self.y = labels
        self.sample_index = sample_index
        if not torch.is_tensor(self.x):
            self.x = torch.from_numpy(self.x.astype(np.float32))
        if not torch.is_tensor(self.y):
            self.y = torch.from_numpy(self.y.astype(np.float32))

    def __len__(self,) -> int:
        """Return number of examples in dataset

        Args:

        Returns:
            int: n examples
        """
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        """ Get tuple of features, labels by index

        Args:
            index (int): index

        Returns:
            Tuple[torch.tensor, torch.tensor]:
        """
        return (
            self.x[index, :],
            self.y[index, :],
        )



class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        feature_columns: List[str],
        label_columns: List[str],
        sample_idx_column: str = None,
        batch_size: Optional[int] = None,
        num_workers: int = 10,
        shuffle: bool = True,
        feature_scaler: str = "StandardScaler",
        label_scaler: str = "LogScaler",
    ):
        """ Create a data loader from a csv file and batch samples by batch_size

        Args:
            feature_columns (List[str]):  list of column names to be used as feature
            label_columns (List[str]): list of column names to be used as labels
            batch_size (Optional[int]): size of batch 
            num_workers (int): num_workers
            shuffle (bool): whether to shuffle the dataset 
        """
        super().__init__()
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.sample_idx_column = sample_idx_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.feature_scaler = getattr(scalers, feature_scaler)()
        self.label_scaler = getattr(scalers, label_scaler)()


    def load_dataset(self, df: pd.DataFrame,) -> "Dataset":
        """ Given a dataframe

        Args:
            path (Path): path to csv

        Returns:
            "Dataset"
        """
        x, y, sample_index = read_data(
            df,
            feature_columns=self.feature_columns,
            label_columns=self.label_columns,
            sample_idx_column=self.sample_idx_column,
        )
        return Dataset(features=x, labels=y, sample_index=sample_index,)

    def load_dataset_from_array(self, array) -> "Dataset":
        """ Given a dataframe

        Args:
            path (Path): path to csv

        Returns:
            "Dataset"
        """
        return Dataset(features=array, labels=array)


    def setup(
        self,
        train_data_path: Path,
        test_data_path: Path,
        val_data_path: Optional[Path] = None,
        autoencoder: bool = False
    ):
        """ Setup the training/validation/test datasets

        Args:
            train_data_path (Path): train_data_path
            val_data_path (Path): val_data_path
            test_data_path (Path): test_data_path
        """
        if autoencoder:
            self.train_data= self.load_dataset_from_array(
                array=np.load(train_data_path),
            )
            self.val_data= self.load_dataset_from_array(
                array=np.load(val_data_path),
            )
            self.test_data= self.load_dataset_from_array(
                array=np.load(test_data_path),
            )


        else:
            self.train_data = self.load_dataset(df=pd.read_csv(train_data_path,))
            if val_data_path:
                self.val_data = self.load_dataset(df=pd.read_csv(val_data_path,))
            self.test_data = self.load_dataset(df=pd.read_csv(test_data_path,))

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [
            split for split in KFold(num_folds).split(range(len(self.train_data)))
        ]

    def setup_fold_index(self, fold_index: int):
        train_indices, val_indices = self.splits[fold_index]
        if self.train_data.sample_index is not None:
            train_indices = [
                i
                for i, e in enumerate(self.train_data.sample_index)
                if e in train_indices
            ]
            val_indices = [
                i
                for i, e in enumerate(self.train_data.sample_index)
                if e in val_indices
            ]
        self.train_fold = Subset(self.train_data, train_indices)
        self.val_fold = Subset(self.train_data, val_indices)

    def transform_datasets(self,):
        self.train_data.x = self.feature_scaler.fit_transform(self.train_data.x)
        self.test_data.x = self.feature_scaler.transform(self.test_data.x)
        self.train_data.y = self.label_scaler.fit_transform(self.train_data.y)
        self.test_data.y = self.label_scaler.transform(self.test_data.y)
        if hasattr(self, "val_data"):
            self.val_data.x = self.feature_scaler.transform(self.val_data.x)
            self.val_data.y = self.label_scaler.transform(self.val_data.y)

    def train_dataloader(self,) -> torch.utils.data.DataLoader:
        """train_dataloader.

        Args:

        Returns:
            torch.utils.data.DataLoader:
        """
        return torch.utils.data.DataLoader(
            self.train_fold if hasattr(self, "train_fold") else self.train_data,
            batch_size=self.batch_size
            if self.batch_size is not None
            else len(self.train_data),
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """val_dataloader.

        Args:

        Returns:
            torch.utils.data.DataLoader:
        """
        return torch.utils.data.DataLoader(
            self.val_fold if hasattr(self, "train_fold") else self.val_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size
            if self.batch_size is not None
            else len(self.val_data),
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """test_dataloader.

        Args:

        Returns:
            torch.utils.data.DataLoader:
        """
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.batch_size
            if self.batch_size is not None
            else len(self.test_data),
            num_workers=self.num_workers,
        )

    def dump_scalers(self, path: str):
        """ Dump fitted scalers to pickle file
        to be used for prediction

        Args:
            path (str): path
        """
        scaler_dict = {
            "features": self.feature_scaler.to_dict(),
            "labels": self.label_scaler.to_dict(),
        }
        with open(path, "wb") as fp:
            pickle.dump(scaler_dict, fp)


