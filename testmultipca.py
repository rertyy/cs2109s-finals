import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split

from sklearn.neighbors import NearestNeighbors
import numpy as np
import random


# def get_augmentations():
#     return transforms.Compose([transforms.RandomHorizontalFlip(),
#                                transforms.RandomVerticalFlip(),
#                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#                                ])


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    Copied directly from https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
    """

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def generate_synthetic(X, labels, n_neighbors=3):
    X = X.copy()
    print(X.shape)
    X_where_y0 = X[labels == 0]  # majority class
    X_where_y1 = X[labels == 1]
    X_where_y2 = X[labels == 2]
    y0_num = X_where_y0.shape[0]
    y1_num = X_where_y1.shape[0]
    y2_num = X_where_y2.shape[0]

    X_w_y1_reshaped = X_where_y1.reshape(X_where_y1.shape[0], -1)
    X_w_y2_reshaped = X_where_y2.reshape(X_where_y2.shape[0], -1)

    y1_upsample = y0_num - y1_num
    y2_upsample = y0_num - y2_num

    X_w_y1_synthetic = smote(X_w_y1_reshaped, y1_upsample, n_neighbors)
    X_w_y2_synthetic = smote(X_w_y2_reshaped, y2_upsample, n_neighbors)

    X_w_y1_synthetic = X_w_y1_synthetic.reshape(-1, *X_where_y1.shape[1:])
    X_w_y2_synthetic = X_w_y2_synthetic.reshape(-1, *X_where_y2.shape[1:])

    X_oversampled = np.vstack([X, X_w_y1_synthetic, X_w_y2_synthetic])
    y_oversampled = np.hstack([
        labels,
        np.ones(X_w_y1_synthetic.shape[0]),
        np.full(X_w_y2_synthetic.shape[0], 2)
    ])

    return X_oversampled, y_oversampled


def smote(X, num_oversamples, n_neighbors=5, seed=2109):
    np.random.seed(seed)
    n_samples, n_features = X.shape
    synthetic_samples = np.zeros((num_oversamples, n_features))

    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
    nn.fit(X)

    indices = np.random.randint(0, n_samples, size=num_oversamples)
    samples = X[indices]

    nnres = nn.kneighbors(samples, return_distance=False)

    nn_indices = nnres[np.arange(num_oversamples), np.random.randint(0, n_neighbors, size=num_oversamples)]
    nn_samples = X[nn_indices]

    diffs = nn_samples - samples
    synthetic_samples = samples + diffs * np.random.random(size=(num_oversamples, 1))

    return synthetic_samples.reshape(num_oversamples, *X.shape[1:])


def drop_nan_y(X, y):
    nan_indices = np.argwhere(np.isnan(y)).squeeze()
    mask = np.ones(y.shape, bool)
    mask[nan_indices] = False
    X = X[mask]
    y = y[mask]
    return X, y


def clean_x_data(X):
    X[np.isnan(X)] = np.nanmedian(X)
    X[X < 0] = 0
    X[X > 255] = 255
    # lower = np.percentile(X, 25) * 1.15
    # upper = np.percentile(X, 75) * 1.5
    # X[X < lower] = lower
    # X[X > upper] = upper
    return X

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



# class CustomNeuralNetwork(nn.Module):
#     def __init__(self, input_size, classes=3, drop_prob=0.5):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Flatten(),
#         )
#
#         self.fc = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),  # New fully connected layer
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, classes)
#         )
#
#
#     def forward(self, x):
#         x = self.network(x)
#         # print(x.shape)
#         x = self.fc(x)
#         return x


class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_size, classes=3, drop_prob=0.3, seed=2109):
        super().__init__()
        torch.manual_seed(seed)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, classes)
        )

    def forward(self, x):
        x = self.network(x)
        # print(x.shape)
        x = self.fc(x)
        return x


class Model:
    """
    This class represents an AI model.
    """

    def __init__(self,
                 batch_size=20,
                 epochs=20,  # epochs seem to get worse after about 10 at num_components=256
                 # learning_rate=1e-3,
                 criterion=nn.CrossEntropyLoss,
                 num_components=256,
                 scaler=MinMaxScaler(),
                 learning_rate=0.0003826645125269827,
                 dropout=0.23535222860200122,
                 seed = 2109
                 ):
        """
        Constructor for Model class.

        Parameters
        ----------
        self : object
            The instance of the object passed by Python.
        """
        # TODO: Replace the following code with your own initialization code.
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.optimizer = None
        self.model = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.criterion = criterion()
        self.num_components = num_components
        self.pcas = [PCA(n_components=num_components, svd_solver='full') for _ in range(3)]
        self.scalers = [StandardScaler() for _ in range(3)]


        self.scaler = scaler
        self.dropout = dropout
        self.seed = seed

        self.g = torch.Generator()
        self.g.manual_seed(self.seed)


        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def fit(self, X, y):
        """
        Train the model using the input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, channel, height, width)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of the trained model.
        """
        # TODO: Add your training code.

        self.model = CustomNeuralNetwork(input_size=self.num_components, drop_prob=self.dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print('start')

        X, y = drop_nan_y(X, y)

        X = clean_x_data(X)

        # print("pre-synthetic")
        X, y = generate_synthetic(X, y, 5)
        # print(y.min())

        # X, X_test, y, y_test = train_test_split(X, y, test_size=100)
        # print(y.min())

        reshaped_channels = [X[:, i, :, :].reshape(X.shape[0], -1) for i in range(X.shape[1])]
        scaled_channels = [self.scalers[i].fit_transform(channel.T).T for i, channel in enumerate(reshaped_channels)]
        transformed_channels = [self.pcas[i].fit_transform(channel) for i, channel in enumerate(scaled_channels)]
        reconstructed_channels = [self.pcas[i].inverse_transform(transformed_channel) for i, transformed_channel in enumerate(transformed_channels)]
        reconstructed_image = np.stack([channel.reshape(X.shape[0], 16, 16) for channel in reconstructed_channels], axis=-1)
        reconstructed_image_input = np.transpose(reconstructed_image, (0, 3, 1, 2))
        # reconstructed_image_input = np.expand_dims(reconstructed_image, axis=0)




        pca_result_tensor = torch.tensor(reconstructed_image_input, dtype=torch.float32)  #.to(self.device)
        labels_tensor = torch.tensor(y, dtype=torch.long)  # .to(self.device)

        # print(y.min())
        # dataset = CustomTensorDataset(tensors=(pca_result_tensor, labels_tensor), transform=get_augmentations())
        dataset = TensorDataset(pca_result_tensor, labels_tensor)
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  worker_init_fn=seed_worker,
                                  generator=self.g
                                  )
        # print("pre-epoch")

        epoch_losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0
            # print(f"Epoch {epoch+1}")
            for inputs, labels in train_loader:
                # print(inputs, labels)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_losses.append(epoch_loss / len(train_loader))
            print(f"Epoch {epoch + 1} loss: {epoch_losses[-1]}")

        return self

    def predict(self, X):
        """
        Use the trained model to make predictions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, channel, height, width)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
        Predicted target values per element in X.

        """
        # TODO: Replace the following code with your own prediction code.
        X = clean_x_data(X)

        X = torch.from_numpy(X).float()
        # X.to(self.device)
        self.model.eval()


        reshaped_channels = [X[:, i, :, :].reshape(X.shape[0], -1) for i in range(X.shape[1])]
        scaled_channels = [self.scalers[i].fit_transform(channel.T).T for i, channel in enumerate(reshaped_channels)]
        transformed_channels = [self.pcas[i].fit_transform(channel) for i, channel in enumerate(scaled_channels)]
        reconstructed_channels = [self.pcas[i].inverse_transform(transformed_channel) for i, transformed_channel in enumerate(transformed_channels)]
        reconstructed_image = np.stack([channel.reshape(X.shape[0], 16, 16) for channel in reconstructed_channels], axis=-1)
        reconstructed_image_input = np.transpose(reconstructed_image, (0, 3, 1, 2))
        # reconstructed_image_input = np.expand_dims(reconstructed_image, axis=0)

        print("fit shape:", reconstructed_image_input.shape)

        original_pca = torch.tensor(reconstructed_image_input, dtype=torch.float32)  #.to(self.device)
        with torch.no_grad():
            outputs = self.model(original_pca)
        return outputs.detach().numpy().argmax(axis=1)


#%%
# Import packages
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
with open('data.npy', 'rb') as f:
    data = np.load(f, allow_pickle=True).item()
    X = data['image']
    y = data['label']
# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Filter test data that contains no labels
# In Coursemology, the test data is guaranteed to have labels
nan_indices = np.argwhere(np.isnan(y_test)).squeeze()
mask = np.ones(y_test.shape, bool)
mask[nan_indices] = False
X_test = X_test[mask]
y_test = y_test[mask]


# Train and predict

model = Model()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate model predition
# Learn more: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
print("F1 Score (macro): {0:.2f}".format(f1_score(y_test, y_pred, average='macro'))) # You may encounter errors, you are expected to figure out what's the issue.

#%%

# N fold cross validation
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

with open('data.npy', 'rb') as f:
    data = np.load(f, allow_pickle=True).item()
    X = data['image']
    y = data['label']


nan_indices = np.argwhere(np.isnan(y)).squeeze()
mask = np.ones(y.shape, bool)
mask[nan_indices] = False
X = X[mask]
y = y[mask]

num_folds = 10

model = Model()
kf = KFold(n_splits=num_folds, shuffle=True, random_state=2109)

f1_scores = []

for train_index, test_index in kf.split(X):
    model.fit(X=X[train_index], y=y[train_index])

    predictions = model.predict(X[test_index])

    score = f1_score(y[test_index], predictions, average='macro')

    f1_scores.append(score)
    print("f1:", score)

print("F1:", f1_scores)
print("Mean:", np.mean(f1_scores))
print("Std:", np.std(f1_scores))
print("Max:", np.max(f1_scores))
print("Min:", np.min(f1_scores))
