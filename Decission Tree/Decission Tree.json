{
    "metadata": {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3 (ipykernel)",
            "language": "python"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py"
        }
    },
    "nbformat_minor": 2,
    "nbformat": 4,
    "cells": [
        {
            "cell_type": "code",
            "source": [
                "import pandas as pd\r\n",
                "import numpy as np\r\n",
                "import matplotlib.pyplot as plt\r\n",
                "plt.rcParams.update({'figure.figsize':(7,3), 'figure.dpi':120})"
            ],
            "metadata": {
                "azdata_cell_guid": "156d81f9-4966-44cb-bb6b-65ad71efafcf",
                "language": "python"
            },
            "outputs": [],
            "execution_count": 2
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn.datasets import load_iris\r\n",
                "iris = load_iris()\r\n",
                "dir(iris)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "82a867d9-c7f8-49a6-8cb5-74bd05baf71d"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 3,
                    "data": {
                        "text/plain": "['DESCR',\n 'data',\n 'data_module',\n 'feature_names',\n 'filename',\n 'frame',\n 'target',\n 'target_names']"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 3
        },
        {
            "cell_type": "code",
            "source": [
                "df = pd.DataFrame(iris.data, columns=iris.feature_names)\r\n",
                "df"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "0f717e48-6bbc-40cf-9bfa-34e919c9500a"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 4,
                    "data": {
                        "text/plain": "     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)\n0                  5.1               3.5                1.4               0.2\n1                  4.9               3.0                1.4               0.2\n2                  4.7               3.2                1.3               0.2\n3                  4.6               3.1                1.5               0.2\n4                  5.0               3.6                1.4               0.2\n..                 ...               ...                ...               ...\n145                6.7               3.0                5.2               2.3\n146                6.3               2.5                5.0               1.9\n147                6.5               3.0                5.2               2.0\n148                6.2               3.4                5.4               2.3\n149                5.9               3.0                5.1               1.8\n\n[150 rows x 4 columns]",
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>sepal length (cm)</th>\n      <th>sepal width (cm)</th>\n      <th>petal length (cm)</th>\n      <th>petal width (cm)</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>5.1</td>\n      <td>3.5</td>\n      <td>1.4</td>\n      <td>0.2</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>4.9</td>\n      <td>3.0</td>\n      <td>1.4</td>\n      <td>0.2</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>4.7</td>\n      <td>3.2</td>\n      <td>1.3</td>\n      <td>0.2</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4.6</td>\n      <td>3.1</td>\n      <td>1.5</td>\n      <td>0.2</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>5.0</td>\n      <td>3.6</td>\n      <td>1.4</td>\n      <td>0.2</td>\n    </tr>\n    <tr>\n      <th>...</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>145</th>\n      <td>6.7</td>\n      <td>3.0</td>\n      <td>5.2</td>\n      <td>2.3</td>\n    </tr>\n    <tr>\n      <th>146</th>\n      <td>6.3</td>\n      <td>2.5</td>\n      <td>5.0</td>\n      <td>1.9</td>\n    </tr>\n    <tr>\n      <th>147</th>\n      <td>6.5</td>\n      <td>3.0</td>\n      <td>5.2</td>\n      <td>2.0</td>\n    </tr>\n    <tr>\n      <th>148</th>\n      <td>6.2</td>\n      <td>3.4</td>\n      <td>5.4</td>\n      <td>2.3</td>\n    </tr>\n    <tr>\n      <th>149</th>\n      <td>5.9</td>\n      <td>3.0</td>\n      <td>5.1</td>\n      <td>1.8</td>\n    </tr>\n  </tbody>\n</table>\n<p>150 rows × 4 columns</p>\n</div>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 4
        },
        {
            "cell_type": "code",
            "source": [
                "df['y']= iris.target\r\n",
                "df.sample(5)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "1ab7d443-3a74-4feb-a300-cd9007a11a65"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 6,
                    "data": {
                        "text/plain": "     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n77                 6.7               3.0                5.0               1.7   \n131                7.9               3.8                6.4               2.0   \n49                 5.0               3.3                1.4               0.2   \n130                7.4               2.8                6.1               1.9   \n43                 5.0               3.5                1.6               0.6   \n\n     y  \n77   1  \n131  2  \n49   0  \n130  2  \n43   0  ",
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>sepal length (cm)</th>\n      <th>sepal width (cm)</th>\n      <th>petal length (cm)</th>\n      <th>petal width (cm)</th>\n      <th>y</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>77</th>\n      <td>6.7</td>\n      <td>3.0</td>\n      <td>5.0</td>\n      <td>1.7</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>131</th>\n      <td>7.9</td>\n      <td>3.8</td>\n      <td>6.4</td>\n      <td>2.0</td>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>49</th>\n      <td>5.0</td>\n      <td>3.3</td>\n      <td>1.4</td>\n      <td>0.2</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>130</th>\n      <td>7.4</td>\n      <td>2.8</td>\n      <td>6.1</td>\n      <td>1.9</td>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>43</th>\n      <td>5.0</td>\n      <td>3.5</td>\n      <td>1.6</td>\n      <td>0.6</td>\n      <td>0</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 6
        },
        {
            "cell_type": "code",
            "source": [
                "X = df.drop(['y'],axis='columns')\r\n",
                "y = iris.target"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "0c62e748-1e22-4e1e-94b8-f9a5c36f3e12"
            },
            "outputs": [],
            "execution_count": 8
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn.model_selection import train_test_split\r\n",
                "X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.7, random_state=0)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "d11ec54b-fea1-4feb-94f0-352c723d4bf6"
            },
            "outputs": [],
            "execution_count": 9
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn import tree\r\n",
                "model_tree = tree.DecisionTreeClassifier(max_depth=2)\r\n",
                "model_tree.fit(X_train,y_train)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "50396a03-a4d2-4cb3-bcb5-225c312b999e"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 11,
                    "data": {
                        "text/plain": "DecisionTreeClassifier(max_depth=2)",
                        "text/html": "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier(max_depth=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier(max_depth=2)</pre></div></div></div></div></div>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 11
        },
        {
            "cell_type": "code",
            "source": [
                "y_pred = model_tree.predict(X_test)\r\n",
                "y_pred"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "ff9cdba7-3665-447b-b105-c5a590174259"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 12,
                    "data": {
                        "text/plain": "array([2, 1, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1,\n       0, 0, 2, 0, 0, 1, 1, 0, 2, 1, 0, 2, 2, 1, 0, 1, 1, 1, 2, 0, 2, 0,\n       0, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 0, 2, 1, 1, 1,\n       1, 2, 0, 0, 2, 1, 0, 0, 1, 0, 2, 1, 0, 1, 2, 1, 0, 2, 2, 2, 2, 0,\n       0, 2, 2, 0, 2, 0, 2, 2, 0, 0, 1, 0, 0, 0, 1, 2, 2])"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 12
        },
        {
            "cell_type": "code",
            "source": [
                "y_test"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "53a5f659-a491-485d-b9b3-c9f0e6c0fe43"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 13,
                    "data": {
                        "text/plain": "array([2, 1, 0, 2, 0, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1,\n       0, 0, 2, 0, 0, 1, 1, 0, 2, 1, 0, 2, 2, 1, 0, 1, 1, 1, 2, 0, 2, 0,\n       0, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 0, 2, 1, 1, 1,\n       1, 2, 0, 0, 2, 1, 0, 0, 1, 0, 2, 1, 0, 1, 2, 1, 0, 2, 2, 2, 2, 0,\n       0, 2, 2, 0, 2, 0, 2, 2, 0, 0, 2, 0, 0, 0, 1, 2, 2])"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 13
        },
        {
            "cell_type": "code",
            "source": [
                "import seaborn as sns\r\n",
                "from sklearn import metrics\r\n",
                "cnf_matrix = metrics.confusion_matrix(y_test,y_pred)\r\n",
                "cnf_matrix"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "1e259384-7722-49a0-bbbf-3fc88a26e198"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 14,
                    "data": {
                        "text/plain": "array([[33,  0,  0],\n       [ 0, 34,  0],\n       [ 0,  5, 33]], dtype=int64)"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 14
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\r\n",
                "fig, ax = plt.subplots(figsize=(4,4))\r\n",
                "labels = ['Setosa','Versicolor','Virginia']\r\n",
                "ConfusionMatrixDisplay.from_predictions (y_test, y_pred, display_labels=labels, xticks_rotation=45, ax = ax, colorbar=False, cmap=\"Blues_r\")"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "183a9b43-a53e-44c0-89fa-6381476c67b7"
            },
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "text/plain": "<Figure size 480x480 with 1 Axes>",
                        "image/png": "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\n"
                    },
                    "metadata": {}
                },
                {
                    "output_type": "execute_result",
                    "execution_count": 15,
                    "data": {
                        "text/plain": "<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x21acbf5d370>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 15
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn.metrics import accuracy_score\r\n",
                "accuracy_score(y_test,y_pred)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "dd40a267-b519-4a8c-ae56-01162d5073c3",
                "tags": []
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 16,
                    "data": {
                        "text/plain": "0.9523809523809523"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 16
        }
    ]
}