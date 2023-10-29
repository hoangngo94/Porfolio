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
                "import matplotlib.pyplot as plt"
            ],
            "metadata": {
                "azdata_cell_guid": "41232f75-c7c3-4f45-a518-dc5e02f70644",
                "language": "python"
            },
            "outputs": [],
            "execution_count": 3
        },
        {
            "cell_type": "code",
            "source": [
                "df= pd.read_csv(r'C:\\Users\\ITS\\Desktop\\ML\\Logistic Regression\\Data_Lesson3_a_Insurance.csv')\r\n",
                "df.head()"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "65bd6719-a43e-46d3-85e9-8306558b802d"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 4,
                    "data": {
                        "text/plain": "   Age  Bought_Insurance\n0   22                 0\n1   25                 0\n2   47                 1\n3   52                 0\n4   46                 1",
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Age</th>\n      <th>Bought_Insurance</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>22</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>25</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>47</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>52</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>46</td>\n      <td>1</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 4
        },
        {
            "cell_type": "code",
            "source": [
                "df.isna().sum()"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "c47e44ca-f71a-4ef7-b383-e0fef07e7615"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 5,
                    "data": {
                        "text/plain": "Age                 0\nBought_Insurance    0\ndtype: int64"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 5
        },
        {
            "cell_type": "code",
            "source": [
                "df.describe()"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "f56376ae-cc5a-457f-9a50-fe71cf5fba7d"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 6,
                    "data": {
                        "text/plain": "             Age  Bought_Insurance\ncount  27.000000         27.000000\nmean   39.666667          0.518519\nstd    15.745573          0.509175\nmin    18.000000          0.000000\n25%    25.000000          0.000000\n50%    45.000000          1.000000\n75%    54.500000          1.000000\nmax    62.000000          1.000000",
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Age</th>\n      <th>Bought_Insurance</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>27.000000</td>\n      <td>27.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>39.666667</td>\n      <td>0.518519</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>15.745573</td>\n      <td>0.509175</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>18.000000</td>\n      <td>0.000000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>25.000000</td>\n      <td>0.000000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>45.000000</td>\n      <td>1.000000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>54.500000</td>\n      <td>1.000000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>62.000000</td>\n      <td>1.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 6
        },
        {
            "cell_type": "code",
            "source": [
                "plt.scatter(df.Age, df.Bought_Insurance, color='red')\r\n",
                "plt.xlabel('Age')\r\n",
                "plt.ylabel('Bought_Insurance')\r\n",
                "plt.grid()\r\n",
                "plt.legend()"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "a7485815-cc94-40e1-bd8b-338ce5b3199f"
            },
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stderr",
                    "text": "No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.\n"
                },
                {
                    "output_type": "display_data",
                    "data": {
                        "text/plain": "<Figure size 640x480 with 1 Axes>",
                        "image/png": "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/Lndv+DJ32+/nlCckZGhXr16VRgfNmyYFi9erEceeUSHDh1SRkaGy30mTJigzMxMtWjRQs8++6weeeQRtx8zLy9P9evXd+uEJH9zOBxavXq1+vbtG/jndVjsCsWW6o1VlJbK8dVXWp2Xp76RkQo141VwPWGlK+YGQm+s9Hx7sD63++LP7f7/PPn7bZpPS9UUwg18gd6YE30xL3pjToHUF0/+fgfUOTcAAABXQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWQrgBAACWYopw8/rrr6t169YKDw9Xt27dtHXr1svOnzt3rtq2bas6deooNjZWEyZM0Pnz52uoWgAAYGZ+DzfLli1TcnKypk2bpu+++04dO3ZUUlKSjh8/Xun8JUuWaNKkSZo2bZp++uknLVy4UMuWLdP//M//1HDlAADAjEL8XcCcOXM0atQoDR8+XJI0f/58rVq1SosWLdKkSZMqzN+0aZO6d++uQYMGSZJat26thx56SFu2bKl0/UVFRSoqKnLezsvLkyQ5HA45HA5fb45Plddn9jprI3pjTvTFvOiNOQVSXzyp0a/hpri4WNu3b1dKSopzLCgoSImJidq8eXOl97ntttv017/+VVu3blXXrl114MABrV69WkOGDKl0/syZM5WamlphfO3atYqIiPDNhlSztLQ0f5eAS6A35kRfzIvemFMg9KWwsNDtuX4NNydPnlRpaamioqJcxqOiorR79+5K7zNo0CCdPHlSt99+uwzDUElJiR5//PFLHpZKSUlRcnKy83ZeXp5iY2PVp08fRUZG+m5jqoHD4VBaWpp69+6t0NBQf5eDX6A35kRfzIvemFMg9aX8yIs7/H5YylMZGRmaMWOG3njjDXXr1k379u3T+PHj9ac//UnPPvtshfl2u112u73CeGhoqOkbWS6Qaq1t6I050RfzojfmFAh98aQ+v4abxo0bKzg4WDk5OS7jOTk5io6OrvQ+zz77rIYMGaKRI0dKkjp06KCCggI99thjmjx5soKC/H6ONAAA8CO/JoGwsDB16dJF6enpzrGysjKlp6crISGh0vsUFhZWCDDBwcGSJMMwqq9YAAAQEPx+WCo5OVnDhg1TfHy8unbtqrlz56qgoMD56amhQ4eqefPmmjlzpiTpnnvu0Zw5c9S5c2fnYalnn31W99xzjzPkAACA2svv4WbgwIE6ceKEpk6dquzsbHXq1Elr1qxxnmR8+PBhlz01U6ZMkc1m05QpU/Tzzz+rSZMmuueee/T888/7axMAAICJ+D3cSNK4ceM0bty4SpdlZGS43A4JCdG0adM0bdq0GqgMAAAEGs6+BQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAlkK4AQAAllLlcFNcXKysrCyVlJT4oh4AAIAq8TrcFBYW6tFHH1VERIRuuukmHT58WJL0hz/8QbNmzfJZgQAAAJ7wOtykpKTohx9+UEZGhsLDw53jiYmJWrZsmU+KAwAA8FSIt3dcsWKFli1bpltvvVU2m805ftNNN2n//v0+KQ4AAMBTXu+5OXHihJo2bVphvKCgwCXsAAAA1CSvw018fLxWrVrlvF0eaN566y0lJCRUvTIAAAAveH1YasaMGfr1r3+tzMxMlZSU6JVXXlFmZqY2bdqkDRs2+LJGAAAAt3m95+b222/Xjh07VFJSog4dOmjt2rVq2rSpNm/erC5duviyRgAAALd5vedGkq655hq9+eabvqoFAACgyrzec7N69Wp98cUXFca/+OILff7551UqCgAAwFteh5tJkyaptLS0wrhhGJo0aVKVigIAAPCW1+Fm7969uvHGGyuMt2vXTvv27atSUQAAAN7yOtzUr19fBw4cqDC+b98+1a1bt0pFAQAAeMvrcHPffffpqaeecrka8b59+zRx4kTde++9PikOAADAU16Hm9mzZ6tu3bpq166d2rRpozZt2uiGG27Q1VdfrRdffNGXNQIAALjN64+C169fX5s2bVJaWpp++OEH1alTR3FxcerZs6cv6wMAAPBIla5zY7PZ1KdPH/Xp08dX9QAAAFRJlcJNenq60tPTdfz4cZWVlbksW7RoUZUKAwAA8IbX4SY1NVXPPfec4uPjFRMTwzeBAwAAU/A63MyfP1+LFy/WkCFDfFkPAABAlXj9aani4mLddtttvqwFAACgyrwONyNHjtSSJUt8WQsAAECVeX1Y6vz581qwYIG+/PJLxcXFKTQ01GX5nDlz3F7X66+/rhdeeEHZ2dnq2LGj/vznP6tr166XnJ+bm6vJkyfr448/1qlTp9SqVSvNnTtXffv29XZzAACARXgdbnbu3KlOnTpJkn788UeXZZ6cXLxs2TIlJydr/vz56tatm+bOnaukpCRlZWWpadOmFeYXFxerd+/eatq0qZYvX67mzZvrn//8pxo0aODtpgAAAAvxOtysX7/eJwXMmTNHo0aN0vDhwyVdOFF51apVWrRoUaXfLr5o0SKdOnVKmzZtcu4tat269SXXX1RUpKKiIuftvLw8SZLD4ZDD4fDJNlSX8vrMXmdtRG/Mib6YF70xp0Dqiyc12gzDMKqxlssqLi5WRESEli9frv79+zvHhw0bptzcXH366acV7tO3b181atRIERER+vTTT9WkSRMNGjRIzzzzjIKDgyvMnz59ulJTUyuML1myRBERET7dHgAAUD0KCws1aNAgnTlzRpGRkZedW6WL+G3btk0ffPCBDh8+rOLiYpdlH3/88RXvf/LkSZWWlioqKsplPCoqSrt37670PgcOHNC6des0ePBgrV69Wvv27dMTTzwhh8OhadOmVZifkpKi5ORk5+28vDzFxsaqT58+V3xy/M3hcCgtLU29e/eucE4T/IvemBN9MS96Y06B1JfyIy/u8DrcLF26VEOHDlVSUpLWrl2rPn36aM+ePcrJydFvf/tbb1d7RWVlZWratKkWLFig4OBgdenSRT///LNeeOGF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TZ06VdnZ2erUqZPWrFnjPMn48OHDCgoyxbUGAQBAAPA63IwYMeKyyxctWuT2usaNG6dx48ZVuiwjI+Oy9128eLHbjwMAAKzP63Bz+vRpl9sOh0M//vijcnNznScUAwAA1DSvw80nn3xSYaysrExjxoyp8IklAACAmuLTk1mCgoKUnJysl19+2ZerBQAAcJvPz9Tdv39/pRf3AwAAqAleH5b65VV/JckwDB07dkyrVq3SsGHDqlwYAACAN7wON99//73L7aCgIDVp0kQvvfTSFT9JBQAAUF28Djfr16/3ZR0AAAA+UeWL+J04cUJZWVmSLlyhuEmTJlUuCgAAwFten1BcUFCgESNGKCYmRj179lTPnj3VrFkzPfroox59LTkAAIAveR1ukpOTtWHDBq1cuVK5ubnKzc3Vp59+qg0bNmjixIm+rBEAAMBtXh+W+uijj7R8+XLdeeedzrG+ffuqTp06evDBBzVv3jxf1AcAAOARr/fcFBYWOr/c8peaNm3KYSkAAOA3XoebhIQETZs2TefPn3eOnTt3TqmpqUpISPBJcQAAAJ7y+rDUK6+8oqSkJLVo0UIdO3aUJP3www+y2+1au3atzwoEAADwhNfhpn379tq7d6/ee+897d69W5L00EMPafDgwapTp47PCgQAAPCE14el/v3vfysiIkKjRo3S+PHjVbduXWVlZWnbtm2+rA8AAMAjHoebXbt2qXXr1mratKnatWunHTt2qGvXrnr55Ze1YMEC9erVSytWrKiGUgEAAK7M43Dzxz/+UR06dNBXX32lO++8U7/5zW/Ur18/nTlzRqdPn9bo0aM1a9as6qgVAADgijw+5+bbb7/VunXrFBcXp44dO2rBggV64oknFBR0ISf94Q9/0K233urzQgEAANzh8Z6bU6dOKTo6WpJUr1491a1bVw0bNnQub9iwofLz831XIQAAgAe8OqHYZrNd9jYAAIC/ePVR8EceeUR2u12SdP78eT3++OOqW7euJKmoqMh31QEAAHjI43AzbNgwl9sPP/xwhTlDhw71viIAAIAq8DjcvP3229VRBwAAgE94fRE/AAAAMyLcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAA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DRs2+LusWu3IkSMaP3680tLSFB4e7u9y8Au//vWvnf+Oi4tTt27d1KpVK33wwQeqU6eOHytDWVmZ4uPjNWPGDElS586d9eOPP2r+/PkaNmyYn6vzPfbcmMi4ceP0t7/9TevXr1eLFi2c49HR0SouLlZubq7L/JycHEVHR9dwlbVLWFiYrr32WnXp0kUzZ85Ux44d9corr9ATP9q+fbuOHz+um2++WSEhIQoJCdGGDRv06quvKiQkRFFRUfTGJBo0aKDrr79e+/bt4z3jZzExMbrxxhtdxm644QbnYcPyHlz86bVA7Q/hxgQMw9C4ceP0ySefaN26dWrTpo3L8i5duig0NFTp6enOsaysLB0+fFgJCQk1XW6tVlZWpqKiInriR3fffbd27dqlHTt2OH/i4+M1ePBg57/pjTmcPXtW+/fvV0xMDO8ZP+vevXuFS4zs2bNHrVq1kiS1adNG0dHRLv3Jy8vTli1bArM//j6jGYYxZswYo379+kZGRoZx7Ngx509hYaFzzuOPP260bNnSWLdunbFt2zYjISHBSEhI8GPV1jdp0iRjw4YNxsGDB42dO3cakyZNMmw2m7F27VrDMOiJmfzy01KGQW/8ZeLEiUZGRoZx8OBB45tvvjESExONxo0bG8ePHzcMg77409atW42QkBDj+eefN/bu3Wu89957RkREhPHXv/7VOWfWrFlGgwYNjE8//dTYuXOncd999xlt2rQxzp0758fKvUO4MQFJlf68/fbbzjnnzp0znnjiCaNhw4ZGRESE8dvf/tY4duyY/4quBUaMGGG0atXKCAsLM5o0aWLcfffdzmBjGPTETC4ON/TGPwYOHGjExMQYYWFhRvPmzY2BAwca+/btcy6nL/61cuVKo3379obdbjfatWtnLFiwwGV5WVmZ8eyzzxpRUVGG3W437r77biMrK8tP1VaNzTAMw597jgAAAHyJc24AAIClEG4AAIClEG4AAIClEG4AAIClEG4AAIClEG4AAIClEG4AAIClEG4AAIClEG4AAIClEG4ABITNmzcrODhY/fr183cpAEyOr18AEBBGjhypevXqaeHChcrKylKzZs38XRIAk2LPDQDTO3v2rJYtW6YxY8aoX79+Wrx4scvyzz77TNddd53Cw8PVq1cvvfPOO7LZbMrNzXXO2bhxo3r06KE6deooNjZWTz75pAoKCmp2QwDUCMINANP74IMP1K5dO7Vt21YPP/ywFi1apPKdzgcPHtT999+v/v3764cfftDo0aM1efJkl/vv379f//Vf/6UBAwZo586dWrZsmTZu3Khx48b5Y3MAVDMOSwEwve7du+vBBx/U+PHjVVJSopiYGH344Ye68847NWnSJK1atUq7du1yzp8yZYqef/55nT59Wg0aNNDIkSMVHBysv/zlL845Gzdu1B133KGCggKFh4f7Y7MAVBP23AAwtaysLG3dulUPPfSQJCkkJEQDBw7UwoULnctvueUWl/t07drV5fYPP/ygxYsXq169es6fpKQklZWV6eDBgzWzIQBqTIi/CwCAy1m4cKFKSkpcTiA2DEN2u12vvfaaW+s4e/asRo8erSeffLLCspYtW/qsVgDmQLgBYFolJSV699139dJLL6lPnz4uy/r376/3339fbdu21erVq12Wffvtty63b775ZmVmZuraa6+t9poB+B/n3AAwrRUrVmjgwIE6fvy46tev77LsmWee0bp16/TBBx+obdu2mjBhgh599FHt2LFDEydO1L/+9S/l5uaqfv362rlzp2699VaNGDFCI0eOVN26dZWZmam0tDS39/4ACByccwPAtBYuXKjExMQKwUaSBgwYoG3btik/P1/Lly/Xxx9/rLi4OM2bN8/5aSm73S5JiouL04YNG7Rnzx716NFDnTt31tSpU7lWDmBR7LkBYDnPP/+85s+fryNHjvi7FAB+wDk3AALeG2+8oVtuuUVXX321vvnmG73wwgtcwwaoxQg3AALe3r179b//+786deqUWrZsqYkTJyolJcXfZQHwEw5LAQAAS+GEYgAAYCmEGwAAYCmEGwAAYCmEGwAAYCmEGwAAYCmEGwAAYCmEGwAAYCmEGwAAYCn/D3H5/8yiH7V5AAAAAElFTkSuQmCC\n"
                    },
                    "metadata": {}
                },
                {
                    "output_type": "execute_result",
                    "execution_count": 7,
                    "data": {
                        "text/plain": "<matplotlib.legend.Legend at 0x25c7afc1310>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 7
        },
        {
            "cell_type": "code",
            "source": [
                "X = df[['Age']]\r\n",
                "y = df.Bought_Insurance"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "e683041a-b64c-4051-83d2-46cfc8a070cd"
            },
            "outputs": [],
            "execution_count": 9
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn.model_selection import train_test_split\r\n",
                "X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "7820e335-a883-4eb2-93fb-dbc5c17b189b"
            },
            "outputs": [],
            "execution_count": 13
        },
        {
            "cell_type": "code",
            "source": [
                "X_test"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "c669982a-e581-4d11-a41f-3ab17be5139b"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 14,
                    "data": {
                        "text/plain": "    Age\n2    47\n24   50\n14   49\n17   58\n5    56\n11   28\n23   45\n13   29\n19   18",
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Age</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>2</th>\n      <td>47</td>\n    </tr>\n    <tr>\n      <th>24</th>\n      <td>50</td>\n    </tr>\n    <tr>\n      <th>14</th>\n      <td>49</td>\n    </tr>\n    <tr>\n      <th>17</th>\n      <td>58</td>\n    </tr>\n    <tr>\n      <th>5</th>\n      <td>56</td>\n    </tr>\n    <tr>\n      <th>11</th>\n      <td>28</td>\n    </tr>\n    <tr>\n      <th>23</th>\n      <td>45</td>\n    </tr>\n    <tr>\n      <th>13</th>\n      <td>29</td>\n    </tr>\n    <tr>\n      <th>19</th>\n      <td>18</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 14
        },
        {
            "cell_type": "code",
            "source": [
                "from sklearn.linear_model import LogisticRegression\r\n",
                "model = LogisticRegression()\r\n",
                "model.fit(X_train,y_train)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "ca68c4b9-a96e-4fbf-be27-4c6c033eb76d"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 15,
                    "data": {
                        "text/plain": "LogisticRegression()",
                        "text/html": "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 15
        },
        {
            "cell_type": "code",
            "source": [
                "y_pred = model.predict(X_test)\r\n",
                "y_pred"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "7b44b6f1-2413-4cad-aef3-8b1f44f55b7c"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 19,
                    "data": {
                        "text/plain": "array([1, 1, 1, 1, 1, 0, 1, 0, 0], dtype=int64)"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 19
        },
        {
            "cell_type": "code",
            "source": [
                "y_test"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "a437dae3-b84d-4e37-aab5-b6af95ce26ec"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 18,
                    "data": {
                        "text/plain": "2     1\n24    1\n14    1\n17    1\n5     1\n11    0\n23    1\n13    0\n19    0\nName: Bought_Insurance, dtype: int64"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 18
        },
        {
            "cell_type": "code",
            "source": [
                "model.predict([[60]])"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "97fcd4a4-a673-431f-a8d4-db8b785f2288"
            },
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stderr",
                    "text": "C:\\Users\\ITS\\AppData\\Roaming\\Python\\Python38\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names\n  warnings.warn(\n"
                },
                {
                    "output_type": "execute_result",
                    "execution_count": 21,
                    "data": {
                        "text/plain": "array([1], dtype=int64)"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 21
        },
        {
            "cell_type": "code",
            "source": [
                "model.score(X_test,y_test)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "90a3b881-1594-4c39-83a9-b1db72532017"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 22,
                    "data": {
                        "text/plain": "1.0"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 22
        },
        {
            "cell_type": "code",
            "source": [
                "a= model.coef_\r\n",
                "b= model.intercept_\r\n",
                "a"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "e9380eb6-79da-4fe6-99c8-025074a5df42"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 23,
                    "data": {
                        "text/plain": "array([[0.1015288]])"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 23
        },
        {
            "cell_type": "code",
            "source": [
                "b"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "2cb7a863-65b9-4be8-9a88-f28b05d365f1"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 24,
                    "data": {
                        "text/plain": "array([-4.23087166])"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 24
        },
        {
            "cell_type": "code",
            "source": [
                "model.predict_proba(X_test)"
            ],
            "metadata": {
                "language": "python",
                "azdata_cell_guid": "036acc6f-5318-4d39-b8b7-b092cebff16e"
            },
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 25,
                    "data": {
                        "text/plain": "array([[0.36795916, 0.63204084],\n       [0.3003633 , 0.6996367 ],\n       [0.32212141, 0.67787859],\n       [0.16005769, 0.83994231],\n       [0.18927295, 0.81072705],\n       [0.80028318, 0.19971682],\n       [0.41631374, 0.58368626],\n       [0.78356036, 0.21643964],\n       [0.91708265, 0.08291735]])"
                    },
                    "metadata": {}
                }
            ],
            "execution_count": 25
        }
    ]
}