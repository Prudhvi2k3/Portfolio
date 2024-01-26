{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 206,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.metrics import r2_score\n",
    "from sklearn.preprocessing import LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "df = pd.read_csv('dataset1.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 438 entries, 0 to 437\n",
      "Data columns (total 7 columns):\n",
      " #   Column   Non-Null Count  Dtype  \n",
      "---  ------   --------------  -----  \n",
      " 0   Ranking  438 non-null    int64  \n",
      " 1   Name     438 non-null    object \n",
      " 2   Year     438 non-null    int64  \n",
      " 3   minutes  438 non-null    int64  \n",
      " 4   genre    438 non-null    object \n",
      " 5   Rating   438 non-null    float64\n",
      " 6   Votes    438 non-null    object \n",
      "dtypes: float64(1), int64(3), object(3)\n",
      "memory usage: 24.1+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "No of rows in Dataset is 438\n"
     ]
    }
   ],
   "source": [
    "#preprocessing of dataset\n",
    "df = df.dropna()\n",
    "df = df.drop('Ranking', axis=1)\n",
    "df = df.drop('Votes', axis=1)\n",
    "df = df.drop('Name', axis=1)\n",
    "df = df.drop('Year', axis=1)\n",
    "df[\"genre\"] = LabelEncoder().fit_transform(df[\"genre\"])\n",
    "\n",
    "print(\"No of rows in Dataset is\",len(df))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 210,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split the data into train and test sets\n",
    "X = df.drop('Rating', axis=1)\n",
    "y = df['Rating']\n",
    "           "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 211,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>minutes</th>\n",
       "      <th>genre</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>30</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>100</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>67</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>23</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>78</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>433</th>\n",
       "      <td>110</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>434</th>\n",
       "      <td>24</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435</th>\n",
       "      <td>24</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>436</th>\n",
       "      <td>105</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>437</th>\n",
       "      <td>25</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>438 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     minutes  genre\n",
       "0         30      7\n",
       "1        100      2\n",
       "2         67      0\n",
       "3         23      2\n",
       "4         78      2\n",
       "..       ...    ...\n",
       "433      110      2\n",
       "434       24      0\n",
       "435       24      0\n",
       "436      105      0\n",
       "437       25      5\n",
       "\n",
       "[438 rows x 2 columns]"
      ]
     },
     "execution_count": 211,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0      7.9\n",
       "1      6.8\n",
       "2      8.8\n",
       "3      9.1\n",
       "4      8.3\n",
       "      ... \n",
       "433    7.4\n",
       "434    8.4\n",
       "435    8.4\n",
       "436    7.3\n",
       "437    8.8\n",
       "Name: Rating, Length: 438, dtype: float64"
      ]
     },
     "execution_count": 212,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 213,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Splitting the dataset into Test and Train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 214,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-6\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=5,\n",
       "                      random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" checked><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestRegressor</label><div class=\"sk-toggleable__content\"><pre>RandomForestRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=5,\n",
       "                      random_state=42)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=5,\n",
       "                      random_state=42)"
      ]
     },
     "execution_count": 214,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Initialize and train the Random Forest Regressor model\n",
    "model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 215,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error (MSE): 0.07804090121439326\n"
     ]
    }
   ],
   "source": [
    "# Evaluate the model on the test set\n",
    "y_pred = model.predict(X_test)\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "print(f\"Mean Squared Error (MSE): {mse}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 216,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Root Mean Squared Error: 0.27935801619855705\n"
     ]
    }
   ],
   "source": [
    "rmse = mse**0.5\n",
    "print(f\"Root Mean Squared Error: {rmse}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error (MAE): 0.13160656394336856\n"
     ]
    }
   ],
   "source": [
    "mae = mean_absolute_error(y_test, y_pred)\n",
    "print(f\"Mean Absolute Error (MAE): {mae}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 218,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R-squared (R²): 0.8874648318245574\n"
     ]
    }
   ],
   "source": [
    "r2 = r2_score(y_test, y_pred)\n",
    "print(f\"R-squared (R²): {r2}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 219,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAArMAAAIjCAYAAAAQgZNYAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAABaw0lEQVR4nO3deXxU1f3/8feQbSYhMwkJCQmEEAiyBlAUZRFQURT0K0tRqQtL9WfdEFQqqBWQslmxWjdUKChQFQVcq8giapCyFFGoCASBsGtiksm+3t8fNtOOk22SSSaTvJ6PRx51zrn3zufejOO7J+eeazIMwxAAAADgg1p4uwAAAACgtgizAAAA8FmEWQAAAPgswiwAAAB8FmEWAAAAPoswCwAAAJ9FmAUAAIDPIswCAADAZxFmAQAA4LMIswB8lslk0uzZs71dRqM0e/ZsmUwmp7YOHTpo4sSJ3imoAhXV2BgMHTpUQ4cO9XYZAGqIMAtAkvTiiy/KZDLp4osvrvUxTp8+rdmzZ2vv3r2eK8xHmUwmx0+LFi0UGxurq666Slu3bvV2aW5pDL/TiRMnOl3PoKAgnXfeeXr88cdVUFBQq2N+9913mj17to4dO+bZYgE0OH9vFwCgcVi9erU6dOignTt3KiUlRYmJiW4f4/Tp05ozZ446dOigPn36eL5IH3PllVfqtttuk2EYOnr0qF588UVdfvnl+uijj3TNNdc0eD0HDx5UixbujWE0lt9pUFCQli5dKknKysrSe++9p7lz5+rIkSNavXq128f77rvvNGfOHA0dOlQdOnRw6vv00089UTKABsLILAAdPXpUX331lZ5++mm1bt26VuEArs477zzdcsstuvXWW/X4449r48aNMgxDzzzzTKX7FBQUqKysrF7qCQoKUkBAQL0cu775+/vrlltu0S233KJ77rlHGzZs0CWXXKI33nhD586d8+h7BQYGKjAw0KPHBFB/CLMAtHr1aoWHh2vkyJH6zW9+U2mYzczM1LRp09ShQwcFBQWpXbt2uu2225SWlqatW7fqoosukiRNmjTJ8SfhFStWSKp8vuav5ycWFRXp8ccfV9++fWWz2RQSEqJLL71Un332mdvnde7cOfn7+2vOnDkufQcPHpTJZNLzzz8vSSouLtacOXPUuXNnmc1mRUREaNCgQdq4caPb71uZpKQkRUZG6ujRo5KkrVu3ymQy6c0339Rjjz2mtm3bKjg4WHa7XZK0Y8cOXX311bLZbAoODtaQIUO0bds2l+MmJyfroosuktlsVqdOnfTyyy9X+P4V/Q7q8jutjxprymQyadCgQTIMQz/88IOj/fjx47r77rvVpUsXWSwWRUREaNy4cU7TCVasWKFx48ZJki677DLHeZVPAfn1Z7L897RmzRrNmzdP7dq1k9ls1hVXXKGUlBSX2l544QV17NhRFotF/fr105dfflnhPNznnntOPXr0UHBwsMLDw3XhhRfq73//e52uC9AcMc0AgFavXq0xY8YoMDBQ48eP10svvaRdu3Y5gowk5eTk6NJLL9WBAwc0efJkXXDBBUpLS9P777+vkydPqlu3bnriiSf0+OOP6//9v/+nSy+9VJI0YMAAt2qx2+1aunSpxo8frzvuuEPZ2dlatmyZhg8frp07d7r1p+7o6GgNGTJEa9as0axZs5z63nrrLfn5+TlCzezZs7VgwQLdfvvt6tevn+x2u3bv3q09e/boyiuvdOscKpORkaGMjAyXKRxz585VYGCgHnroIRUWFiowMFBbtmzRNddco759+2rWrFlq0aKFli9frssvv1xffvml+vXrJ0nat2+frrrqKrVu3VqzZ89WSUmJZs2apejo6GrrqevvtCFqrEp5QA0PD3e07dq1S1999ZVuuukmtWvXTseOHdNLL72koUOH6rvvvlNwcLAGDx6sKVOm6K9//aseeeQRdevWTZIc/1uZhQsXqkWLFnrooYeUlZWlJ598UjfffLN27Njh2Oall17Svffeq0svvVTTpk3TsWPHNGrUKIWHh6tdu3aO7V599VVNmTJFv/nNb3T//feroKBA3377rXbs2KHf/va3dbouQLNjAGjWdu/ebUgyNm7caBiGYZSVlRnt2rUz7r//fqftHn/8cUOSsW7dOpdjlJWVGYZhGLt27TIkGcuXL3fZJj4+3pgwYYJL+5AhQ4whQ4Y4XpeUlBiFhYVO22RkZBjR0dHG5MmTndolGbNmzary/F5++WVDkrFv3z6n9u7duxuXX36543Xv3r2NkSNHVnksd0gyfve73xk//fST8eOPPxo7duwwrrjiCkOSsXjxYsMwDOOzzz4zJBkdO3Y08vLyHPuWlZUZnTt3NoYPH+64toZhGHl5eUZCQoJx5ZVXOtpGjRplmM1m4/jx44627777zvDz8zN+/RX/699BXX6n9VVjRSZMmGCEhIQYP/30k/HTTz8ZKSkpxlNPPWWYTCajZ8+eLu//a9u3bzckGa+//rqj7e233zYkGZ999pnL9r/+TJb/nrp16+b02Xz22WedPluFhYVGRESEcdFFFxnFxcWO7VasWGFIcjrm9ddfb/To0aPacwdQPaYZAM3c6tWrFR0drcsuu0zSL3++vfHGG/Xmm2+qtLTUsd3atWvVu3dvjR492uUYnlxeyc/PzzFfsaysTD///LNKSkp04YUXas+ePW4fb8yYMfL399dbb73laNu/f7++++473XjjjY62sLAw/fvf/9bhw4frfhL/sWzZMrVu3VpRUVG6+OKLtW3bNj3wwAOaOnWq03YTJkyQxWJxvN67d68OHz6s3/72t0pPT1daWprS0tKUm5urK664Ql988YXKyspUWlqqDRs2aNSoUWrfvr1j/27dumn48OHV1leX32lD1VguNzdXrVu3VuvWrZWYmKiHHnpIAwcO1HvvvedU6/9ex+LiYqWnpysxMVFhYWG1+vz8r0mTJjnNpS0fqS6f5rB7926lp6frjjvukL//f//wefPNNzuNHku/fN5OnjypXbt21akmAMyZBZq10tJSvfnmm7rssst09OhRpaSkKCUlRRdffLHOnTunzZs3O7Y9cuSIevbs2SB1vfbaa+rVq5dj7mrr1q310UcfKSsry+1jRUZG6oorrtCaNWscbW+99Zb8/f01ZswYR9sTTzyhzMxMnXfeeUpKStL06dP17bff1uk8rr/+em3cuFGbNm3Sjh07lJaWpsWLF7usKJCQkOD0ujxQT5gwwRHgyn+WLl2qwsJCZWVl6aefflJ+fr46d+7s8t5dunSptr66/E4bqsZyZrNZGzdu1MaNG7V8+XJ169ZNP/74o1N4laT8/Hw9/vjjiouLU1BQkCIjI9W6dWtlZmbW6vPzv/43jEv/nd6QkZEh6Zf5upJcppH4+/u7rJjw8MMPq2XLlurXr586d+6se+65p8K5xgCqx5xZoBnbsmWLzpw5ozfffFNvvvmmS//q1at11VVXeeS9KhvpKy0tlZ+fn+P1qlWrNHHiRI0aNUrTp09XVFSU/Pz8tGDBAh05cqRW733TTTdp0qRJ2rt3r/r06aM1a9boiiuuUGRkpGObwYMH68iRI3rvvff06aefaunSpfrL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",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8, 6))\n",
    "sns.scatterplot(x=y_test, y=y_pred)\n",
    "plt.xlabel(\"Actual Ratings\")\n",
    "plt.ylabel(\"Predicted Ratings\")\n",
    "plt.title(\"Actual vs. Predicted Ratings\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 223,
   "metadata": {},
   "outputs": [],
   "source": [
    "user_minutes=90\n",
    "user_genre='Action'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 227,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Rating: 9.352625000000016\n"
     ]
    }
   ],
   "source": [
    "user_input = pd.DataFrame({'minutes': [user_minutes],\n",
    "                            'genre': [user_genre],\n",
    "                            })\n",
    "\n",
    "user_input_encoded = pd.get_dummies(user_input, columns=['minutes', 'genre'], drop_first=True)\n",
    "\n",
    "user_input_encoded = user_input_encoded.reindex(columns=X_train.columns, fill_value=0)\n",
    "\n",
    "predicted_rating = model.predict(user_input_encoded)\n",
    "print(f\"Predicted Rating: {predicted_rating[0]}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.8"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
