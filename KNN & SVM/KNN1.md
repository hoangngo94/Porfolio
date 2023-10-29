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
                "import seaborn as sns"
            ],
            "metadata": {
                "azdata_cell_guid": "55ccb69a-3449-4fbc-830a-38ca779c6737",
                "language": "python"
            },
            "outputs": [],
            "execution_count": 56
        },
        {
            "cell_type": "code",
            "source": [
                "df= pd.read_csv(r'C:\\Users\\ITS\\Desktop\\ML\\KNN & SVM\\Lesson5_a_Weather.csv')\r\n",
                "df.head()"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "1911b0df-7fce-4e03-a4da-24f408dfcb83"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 57,
                    "data": {
                        "text/plain": "   id   outlook temperature humidity    wind play\n0   1     sunny         hot     high    weak   no\n1   2     sunny         hot     high  strong   no\n2   3  overcast         hot     high    weak  yes\n3   4     rainy        mild     high    weak  yes\n4   5     rainy        cool   normal    weak  yes",
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>outlook</th>\n      <th>temperature</th>\n      <th>humidity</th>\n      <th>wind</th>\n      <th>play</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>sunny</td>\n      <td>hot</td>\n      <td>high</td>\n      <td>weak</td>\n      <td>no</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>sunny</td>\n      <td>hot</td>\n      <td>high</td>\n      <td>strong</td>\n      <td>no</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>overcast</td>\n      <td>hot</td>\n      <td>high</td>\n      <td>weak</td>\n      <td>yes</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>rainy</td>\n      <td>mild</td>\n      <td>high</td>\n      <td>weak</td>\n      <td>yes</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>5</td>\n      <td>rainy</td>\n      <td>cool</td>\n      <td>normal</td>\n      <td>weak</td>\n      <td>yes</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 57
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn.preprocessing import LabelEncoder\r\n",
                "df = df.apply(LabelEncoder().fit_transform)\r\n",
                "df.head()"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "f72c3b85-35e4-4f41-b61f-b60d92420170"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 58,
                    "data": {
                        "text/plain": "   id  outlook  temperature  humidity  wind  play\n0   0        2            1         0     1     0\n1   1        2            1         0     0     0\n2   2        0            1         0     1     1\n3   3        1            2         0     1     1\n4   4        1            0         1     1     1",
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>id</th>\n      <th>outlook</th>\n      <th>temperature</th>\n      <th>humidity</th>\n      <th>wind</th>\n      <th>play</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>0</td>\n      <td>2</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>1</td>\n      <td>2</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>2</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>3</td>\n      <td>1</td>\n      <td>2</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>4</td>\n      <td>1</td>\n      <td>0</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 58
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn.model_selection import train_test_split\r\n",
                "y = df[\"play\"].values\r\n",
                "X = X = df.iloc[:, 1:-1].values\r\n",
                "X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.6,random_state=0)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "b273247d-9309-4f43-a04c-cc5b081469ed"
            },
            "outputs": [],
            "execution_count": 59
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn.neighbors import KNeighborsClassifier\r\n",
                "model_knn = KNeighborsClassifier(n_neighbors=5)\r\n",
                "model_knn.fit(X_train,y_train)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "0ff04407-25ea-4222-b575-deeddf48f310"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 60,
                    "data": {
                        "text/plain": "KNeighborsClassifier()",
                        "text/html": "<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-6\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>KNeighborsClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" checked><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier()</pre></div></div></div></div></div>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 60
        },
        {
            "cell_type": "code",
            "source": [
                "y_pred = model_knn.predict(X_test)\r\n",
                "model_knn.score(X_test,y_test)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "dae29d07-0b54-4402-9009-c072cafb1c81"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 61,
                    "data": {
                        "text/plain": "0.6666666666666666"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 61
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn import metrics\r\n",
                "cnf_matrix = metrics. confusion_matrix(y_test,y_pred)\r\n",
                "cnf_matrix"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "7ac631cc-f5e5-4240-8e4e-47b9f2fe9b34"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 62,
                    "data": {
                        "text/plain": "array([[0, 3],\n       [0, 6]], dtype=int64)"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 62
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay\r\n",
                "fig, ax = plt.subplots(figsize=(4, 4))\r\n",
                "labels = ['No','Yes']\r\n",
                "ConfusionMatrixDisplay.from_predictions (y_test, y_pred, display_labels=labels,xticks_rotation=45,ax=ax,colorbar=False,cmap='Blues_r')"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "b06ab153-ad0e-4adf-a516-e76f53eb1915"
            },
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "text/plain": "<Figure size 400x400 with 1 Axes>",
                        "image/png": "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VX5+fpo4caL8/PzcPQo8GD9nJcu4E7QAYCL27AHAAMQeAAxA7AHAAMQeAAxA7AHAAMQeAAxA7AGUKtffDc5VQ12L2AMoNSzLks1m04YNG9SnTx/t3bvX3SN5DGKPO8bf48HVbDab1qxZo379+umRRx7hZ8yF+Ata/Krre1tpaWk6duyYgoODVb16dYWFhclut//qpVWB4srIyFCHDh00duxYDRs2zLE8NTVVNWrUUPny5d043b2N31Lc1vXQr169Wh07dlR8fLz69Omj9u3ba9euXfLy8pLdbnf3mPAQZ86ckb+/vwYPHqzz589r3rx56tChgxo1aqT4+Hilp6e7e8R7FrHHLRUUFMhms+nLL7/UM888o4SEBO3YsUNLly5V8+bN1bFjR+3evVteXl683IZL1KpVS4cPH1bv3r3VqlUrffbZZ2rTpo3+9a9/adWqVUpOTnb3iPcs4z68BL/uhx9+UI0aNeTj46PCwkIdOHBAzZs3V3x8vLy8vHTfffcpKipKdrtdw4cP13//93+rUqVK7h4b95DrOwc2m03Hjx+XzWbTpUuXVL9+fW3atEnz58/X7373Ow0aNEg1atSQt7e32rVrp4KCAjdPfu8i9igiNzdX/fr10+nTp3X48GF5e3srOztb+/btU3Z2toKDg2VZlsLCwvT000/r+eef17lz54g9ii0rK0shISGSpHXr1mnSpEkqLCzUmTNnFBcXpz/96U9avnx5ka/5y1/+ou+++05t2rRxx8gegcM4KMLX11czZ86Uv7+/mjZtKsuy1L17d1WrVk2LFy/WhQsXHJ/dW7duXZUpU4b3Q6PYzp07p4ceekipqanavHmzBgwYoOeff16fffaZpk2bppkzZ2rbtm2OPf+kpCQ99dRTWrRokT766CPVrFnTvd/APYzYowibzaZWrVopMTFRP//8sx5++GHVrl1bPXv21OLFi5WYmKgff/xROTk5WrRokby8vPgFRLFdunRJNptNeXl5+vzzzzV06FA9++yzunz5smbMmKGhQ4fqySefdOxQVKhQQeHh4dq0aZOaNGni5unvbbz1Ejp9+rSOHDmiRx55xLEsPz9fe/fuVb9+/RQREaGtW7dqwoQJWrt2rdLT09W4cWNlZGTo008/5ZcQTmnevLm6d++uzZs368knn9SwYcN0//3364knntDbb78tm82m2bNnq1mz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\n"
                    },
                    "metadata": {}
                },
                {
                    "output_type": "execute_result",
                    "execution_count": 63,
                    "data": {
                        "text/plain": "<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2492bfc4f10>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 63
        },
        {
            "cell_type": "code",
            "source": [
                ""
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "78b15993-a5a4-4c96-a189-c6f522187ee5"
            },
            "outputs": [],
            "execution_count": 63
        }
    ]
}