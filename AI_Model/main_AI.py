from RF_AI import RF_model
from LSTM_AI import LSTM_model
from TRANSFORMER_AI import transformer_AI

from Model_Load import model_load

# [RF_model,LSTM_model,transformer_AI]
def AI_pipeline(model_func):
    df_model=model_func()
    model_load(df_model)

for i in [RF_model,LSTM_model,transformer_AI] :
    print(i)
    AI_pipeline(i)
    print(f"{i} : load in DB")

