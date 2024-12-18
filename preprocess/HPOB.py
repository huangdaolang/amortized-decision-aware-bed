import numpy as np
import torch
from attrdict import AttrDict
import matplotlib.pyplot as plt
import os
import json


class HPOB(object):
    def __init__(self, meta_dataset="glmnet"):
        self.datasets_list = {"ranger": "7609", "glmnet": "5860", "svm": "5891", "rpart": "5859", "xgboost": "5971"}
        self.meta_dataset = meta_dataset
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.data = self.get_data(meta_dataset)
        self.dataset_ids = list(self.data.keys())
        self.n_dataset = len(self.dataset_ids)
        self.min_data_size = len(self.data[self.dataset_ids[0]]['X'])
        self.dim_x = len(self.data[self.dataset_ids[0]]['X'][0])

    def sample(self, batch_size=16, n_context=None, n_query=None, n_target=10, min_n_context=5, max_n_context=10):
        batch = AttrDict()
        n_context = n_context or torch.randint(low=min_n_context, high=max_n_context, size=[1]).item()
        n_query = n_query or self.min_data_size - n_context - n_target
        assert n_target is not None, "n_target should be specified."

        batch.context_x = torch.zeros([batch_size, n_context, self.dim_x])
        batch.context_y = torch.zeros([batch_size, n_context, 1])
        batch.query_x = torch.zeros([batch_size, n_query, self.dim_x])
        batch.query_y = torch.zeros([batch_size, n_query, 1])
        batch.target_x = torch.zeros([batch_size, n_target, self.dim_x])
        batch.target_y = torch.zeros([batch_size, n_target, 1])

        for i in range(batch_size):
            dataset_id = np.random.choice(self.dataset_ids)
            dataset = self.data[dataset_id]
            X = torch.tensor(dataset['X'])
            y = torch.tensor(dataset['y'])
            n_data = X.shape[0]
            indices = torch.randperm(n_data)
            context_indices = indices[:n_context]
            query_indices = indices[n_context:n_context+n_query]
            target_indices = indices[n_context+n_query:n_context+n_query+n_target]

            batch.context_x[i] = X[context_indices]
            batch.context_y[i] = y[context_indices]
            batch.query_x[i] = X[query_indices]
            batch.query_y[i] = y[query_indices]
            batch.target_x[i] = X[target_indices]
            batch.target_y[i] = y[target_indices]

        return batch

    def get_test_set(self):
        with open(f'{self.path}/../data/HPOB/{self.meta_dataset}_test.json', 'r') as f:
            data = json.load(f)
        return data

    def get_bo_initializations(self):
        with open(f'{self.path}/../data/HPOB/bo-initializations.json', 'r') as f:
            data = json.load(f)
        return data

    def get_available_test_set_id(self, data):
        return list(data.keys())

    def sample_test_set(self, n_context=None, n_query=None):
        seeds = ["test0", "test1", "test2", "test3", "test4"]
        all_bo_initializations = self.get_bo_initializations()
        all_dataset_ids = self.get_available_test_set_id(self.get_test_set())
        batch_size = 5 * len(all_dataset_ids)

        batch = AttrDict()
        assert n_context is not None, "n_context should be specified."
        n_query = n_query

        data = self.get_test_set()

        batch.context_x = torch.zeros([batch_size, n_context, self.dim_x])
        batch.context_y = torch.zeros([batch_size, n_context, 1])
        batch.query_x = torch.zeros([batch_size, n_query, self.dim_x])
        batch.query_y = torch.zeros([batch_size, n_query, 1])
        batch.target_x = torch.zeros([batch_size, 1, self.dim_x])
        batch.target_y = torch.zeros([batch_size, 1, 1])

        batch_context_x = []
        batch_context_y = []
        batch_query_x = []
        batch_query_y = []
        batch_target_x = []
        batch_target_y = []

        for dataset_id in all_dataset_ids:
            dataset = data[dataset_id]
            for seed in seeds:
                init_ids = all_bo_initializations[self.datasets_list[self.meta_dataset]][dataset_id][seed]
                X = torch.tensor(dataset['X'])
                y = torch.tensor(dataset['y'])
                x_context = X[init_ids]
                y_context = y[init_ids]

                mask = torch.ones(X.shape[0], dtype=torch.bool)
                mask[init_ids] = False

                X_rest = X[mask]
                y_rest = y[mask]

                x_query = X_rest[:n_query]
                y_query = y_rest[:n_query]
                x_target = X_rest[-1:]
                y_target = y_rest[-1:]

                batch_context_x.append(x_context)
                batch_context_y.append(y_context)
                batch_query_x.append(x_query)
                batch_query_y.append(y_query)
                batch_target_x.append(x_target)
                batch_target_y.append(y_target)

        batch.context_x = torch.stack(batch_context_x, dim=0)
        batch.context_y = torch.stack(batch_context_y, dim=0)
        batch.query_x = torch.stack(batch_query_x, dim=0)
        batch.query_y = torch.stack(batch_query_y, dim=0)
        batch.target_x = torch.stack(batch_target_x, dim=0)
        batch.target_y = torch.stack(batch_target_y, dim=0)

        return batch

    def get_data(self, meta_dataset):
        with open(f'{self.path}/../data/HPOB/{meta_dataset}.json', 'r') as f:
            data = json.load(f)
        return data

    def preprocess_meta_set(self, meta_dataset_name):
        hpob_hdlr = HPOBHandler(root_dir=f"{self.path}/../data/HPOB/", mode="v3")
        meta_dataset_id = self.datasets_list[meta_dataset_name]
        meta_dataset = hpob_hdlr.meta_train_data[meta_dataset_id]

        with open(f'{self.path}/../data/HPOB/{meta_dataset_name}.json', 'w') as f:
            json.dump(meta_dataset, f, indent=4)


class HPOBHandler:

    def __init__(self, root_dir="HPOB/", mode="v3-test", surrogates_dir="saved-surrogates/"):

        """
        Constructor for the HPOBHandler.
        Inputs:
            * root_dir: path to directory with the benchmark data.
            * mode: mode name indicating how to load the data. Options:
                - v1: Loads HPO-B-v1
                - v2: Loads HPO-B-v2
                - v3: Loads HPO-B-v3
                - v3-test: Loads only the meta-test split from HPO-B-v3
                - v3-train-augmented: Loads all splits from HPO-B-v3, but augmenting the meta-train data with the less frequent search-spaces.
            * surrogates_dir: path to directory with surrogates models.

        """

        print("Loading HPO-B handler")
        self.mode = mode
        self.surrogates_dir = surrogates_dir
        self.seeds = ["test0", "test1", "test2", "test3", "test4"]

        if self.mode == "v3-test":
            self.load_data(root_dir, only_test=True)
        elif self.mode == "v3-train-augmented":
            self.load_data(root_dir, only_test=False, augmented_train=True)
        elif self.mode in ["v1", "v2", "v3"]:
            self.load_data(root_dir, version=self.mode, only_test=False)
        else:
            raise ValueError("Provide a valid mode")

        surrogates_file = surrogates_dir+"summary-stats.json"
        if os.path.isfile(surrogates_file):
            with open(surrogates_file) as f:
                self.surrogates_stats = json.load(f)

    def load_data(self, rootdir="", version="v3", only_test=True, augmented_train=False):

        """
        Loads data with some specifications.
        Inputs:
            * root_dir: path to directory with the benchmark data.
            * version: name indicating what HPOB version to use. Options: v1, v2, v3).
            * Only test: Whether to load only testing data (valid only for version v3).  Options: True/False
            * augmented_train: Whether to load the augmented train data (valid only for version v3). Options: True/False

        """

        print("Loading data...")
        meta_train_augmented_path = os.path.join(rootdir, "meta-train-dataset-augmented.json")
        meta_train_path = os.path.join(rootdir, "meta-train-dataset.json")
        meta_test_path = os.path.join(rootdir,"meta-test-dataset.json")
        meta_validation_path = os.path.join(rootdir, "meta-validation-dataset.json")
        bo_initializations_path = os.path.join(rootdir, "bo-initializations.json")

        with open(meta_test_path, "rb") as f:
            self.meta_test_data = json.load(f)

        with open(bo_initializations_path, "rb") as f:
            self.bo_initializations = json.load(f)

        if not only_test:
            if augmented_train or version=="v1":
                with open(meta_train_augmented_path, "rb") as f:
                    self.meta_train_data = json.load(f)
            else:
                with open(meta_train_path, "rb") as f:
                    self.meta_train_data = json.load(f)
            with open(meta_validation_path, "rb") as f:
                self.meta_validation_data = json.load(f)

        if version != "v3":
            temp_data = {}
            for search_space in self.meta_train_data.keys():
                temp_data[search_space] = {}

                for dataset in self.meta_train_data[search_space].keys():
                    temp_data[search_space][dataset] =  self.meta_train_data[search_space][dataset]

                if search_space in self.meta_test_data.keys():
                    for dataset in self.meta_test_data[search_space].keys():
                        temp_data[search_space][dataset] = self.meta_test_data[search_space][dataset]

                    for dataset in self.meta_validation_data[search_space].keys():
                        temp_data[search_space][dataset] = self.meta_validation_data[search_space][dataset]

            self.meta_train_data = None
            self.meta_validation_data = None
            self.meta_test_data = temp_data

        self.search_space_dims = {}

        for search_space in self.meta_test_data.keys():
            dataset = list(self.meta_test_data[search_space].keys())[0]
            X = self.meta_test_data[search_space][dataset]["X"][0]
            self.search_space_dims[search_space] = len(X)


    def normalize(self, y, y_min = None, y_max=None):

        if y_min is None:
            return (y-np.min(y))/(np.max(y)-np.min(y))
        else:
            return(y-y_min)/(y_max-y_min)

    def evaluate (self, bo_method = None, search_space_id = None, dataset_id = None, seed = None, n_trials = 10):

        """
        Evaluates a method on the benchmark with discretized search-spaces.
        Inputs:
            * bo_method: object to evaluate. It should have a function (class method) named 'observe_and_suggest'.
            * search_space_id: Identifier of the search spaces for the evaluation. Option: see original paper.
            * dataset_id: Identifier of the dataset for the evaluation. Options: see original paper.
            * seed: Identifier of the seed for the evaluation. Options: test0, test1, test2, test3, test4.
            * trails: Number of trials (iterations on the opoitmization).
        Ooutput:
            * a list with the maximumu performance (incumbent) for every trial.

        """

        assert bo_method!=None, "Provide a valid method object for evaluation."
        assert hasattr(bo_method, "observe_and_suggest"), "The provided  object does not have a method called ´observe_and_suggest´"
        assert search_space_id!= None, "Provide a valid search space id. See documentatio for valid obptions."
        assert dataset_id!= None, "Provide a valid dataset_id. See documentation for valid options."
        assert seed!=None, "Provide a valid initialization. Valid options are: test0, test1, test2, test3, test4."

        try:
            X = np.array(self.meta_test_data[search_space_id][dataset_id]["X"])
            y = np.array(self.meta_test_data[search_space_id][dataset_id]["y"])
        except KeyError:
            print(self.meta_test_data.keys())
            raise
        y = self.normalize(y)
        data_size = len(X)

        pending_evaluations = list(range(data_size))
        current_evaluations = []

        init_ids = self.bo_initializations[search_space_id][dataset_id][seed]

        for i in range(len(init_ids)):
            idx = init_ids[i]
            pending_evaluations.remove(idx)
            current_evaluations.append(idx)

        max_accuracy_history = [np.max(y[current_evaluations])]
        for i in range(n_trials):
            idx = bo_method.observe_and_suggest(
                X_obs=X[current_evaluations],
                y_obs=y[current_evaluations],
                X_pen=X[pending_evaluations],
            )
            idx = pending_evaluations[idx]
            pending_evaluations.remove(idx)
            current_evaluations.append(idx)
            max_accuracy_history.append(np.max(y[current_evaluations]))

            if max(y) in max_accuracy_history:
                break

        max_accuracy_history+=[max(y).item()]*(n_trials-i-1)

        return max_accuracy_history

    def get_search_spaces(self):
        return list(self.meta_test_data.keys())

    def get_datasets(self, search_space):
        return list(self.meta_test_data[search_space].keys())

    def get_seeds(self):
        return self.seeds

    def get_search_space_dim(self, search_space):

        return self.search_space_dims[search_space]



if __name__ == "__main__":
    sampler = HPOB(meta_dataset="glmnet")
    batch = sampler.sample(batch_size=2, n_context=5, n_query=None, n_target=10)
    print(batch.query_x.shape, batch.query_y.shape)
