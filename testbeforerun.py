import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split


from sklearn.neighbors import NearestNeighbors
import numpy as np

def generate_synthetic(X, labels, n_neighbors=3):
    X = X.copy()
    print(X.shape)
    X_where_y0 = X[labels == 0] # majority class
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


def smote(X, num_oversamples, n_neighbors=5):
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
    return X

class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_size, classes=3, drop_prob=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),

        )
        self.fc = nn.Sequential(
            nn.Linear(16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.network(x)
        x = self.fc(x)
        return x



class Model:
    """
    This class represents an AI model.
    """

    def __init__(self,
                 batch_size=10,
                 epochs=3,
                 learning_rate=1e-3,
                 criterion=nn.CrossEntropyLoss,
                 num_components = 128,
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
        self.pca = PCA(n_components=num_components, svd_solver='full')
        self.scaler = MinMaxScaler()



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

        self.model = CustomNeuralNetwork(input_size=self.num_components)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print('start')

        X, y = drop_nan_y(X, y)

        X = clean_x_data(X)



        print("pre-synthetic")
        X, y = generate_synthetic(X, y, 5)
        # print(y.min())

        # X, X_test, y, y_test = train_test_split(X, y, test_size=100)
        # print(y.min())

        # Flatten and normalize the data
        flattened_data = X.reshape(X.shape[0], -1)

        normalized_data = self.scaler.fit_transform(flattened_data)
        # print("pre-pca")
        # print(y.min())
        pca_result = self.pca.fit_transform(normalized_data)
        reconstructed = self.pca.inverse_transform(pca_result)
        original_pca = reconstructed.reshape(-1, *X.shape[1:])


        pca_result_tensor = torch.tensor(original_pca, dtype=torch.float32) #.to(self.device)
        labels_tensor = torch.tensor(y, dtype=torch.long) # .to(self.device)

        # print(y.min())

        dataset = TensorDataset(pca_result_tensor, labels_tensor)
        train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        print("pre-epoch")

        epoch_losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0
            print(f"Epoch {epoch+1}")
            for inputs, labels in train_loader:
                # print(inputs, labels)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_losses.append(epoch_loss / len(train_loader))
            print(f"Epoch {epoch+1} loss: {epoch_losses[-1]}")

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

        flattened_data = X.reshape(X.shape[0], -1)
        normalized_data = self.scaler.transform(flattened_data)
        pca_result = self.pca.transform(normalized_data)
        reconstructed = self.pca.inverse_transform(pca_result)
        original_pca = reconstructed.reshape(-1, *X.shape[1:])

        print("fit shape:", pca_result.shape)

        original_pca = torch.tensor(original_pca, dtype=torch.float32) #.to(self.device)
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

num_folds = 5

model = Model()
kf = KFold(n_splits=num_folds, shuffle=True, random_state=2109)

f1_scores = []

for train_index, test_index in kf.split(X):
    model.fit(X=X[train_index], y=y[train_index])

    predictions = model.predict(X[test_index])

    score = f1_score(y[test_index], predictions, average='macro')

    f1_scores.append(score)
    print("train_index:", score)

print("F1:", f1_scores)
print("Mean:", np.mean(f1_scores))
print("Std:", np.std(f1_scores))

#%%
