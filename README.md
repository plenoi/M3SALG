# M3SALG 
M3SALG can be widely used for performing high-throughput identification ALG inhabitors.

## Dependency
The packages that this program depends on are <br> 
`scikit-learn==1.0.2 or higher`. <br>
`jpype1` <br>
`lightgbm==3.3.2` <br>
`joblib==1.2.0` <br>
`xgboost==1.6.2` <br> <br>

You can run following command in terminal.<br>
`pip install scikit-learn==1.0.2` <br>
`pip install jpype1` <br>
`pip install lightgbm==1.2.1` <br>
`pip install xgboost==1.6.2` <br>

## How to use M3SALG 
1. Copy your SMILES file into `./input` and change the name to smiles.csv<br>
2. Run command<br>
`python M3SALG.py`
3. The result including SMILE, label and probability will be saved in `./output/predicted_result.csv`
