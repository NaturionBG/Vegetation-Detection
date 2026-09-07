import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import FixedThresholdClassifier
import joblib
import abc
from typing import Any



class AbstractPredictor(abc.ABC):
  
  @abc.abstractmethod
  def predict(self, *args, **kwargs) -> Any:
    pass


class VegetationDetection(AbstractPredictor):
  '''
  The universal class to predic the presense of vegetation at the moment.
  -
  ---
  The Class is designed to be a Singleton. Only **ONE** instance of this class can exist at a time.
  -
  ---
  
  ***USAGE***:
  -
  + intiate the class, no __init__ parameters are taken.
  + use instance.predict(*args) to predict the value on observations. 
  + **NOTE** that the instance.predict(*args) takes the DOY of your prediction time - 
    this determines the model to be used, and the features that **MUST** be used.
  + instance.predict(*args) takes features either as:
    - **a pd.Series object**: the indices are the string feature names, the values are the feature observations.
    - **a pd.DataFrame object**: the indices are the observation number, the columns are the feature names,
      the values are the feature observations.
  + **IF** a pd.DataFrame is passed, it is **MANDATORY** that it contains an additional DOY column for **EACH** observation.
  + **SEE** the ***instance.predict()*** documentation for what features must be used depending on the DOY.
  ---
  **RETURNS**:
  -
  ---
  + When a pd.Series object is passed, returns a **Scalar**
  + When a pd.DataFrame object is passed, 
    returns a tuple that contains the predictions over the dataframe sorted by DOY & a list of DOY in the exact same order.
  '''
  
  _instance = None
  def __new__(class_, *args, **kwargs):
    if not isinstance(class_._instance, class_):
      class_._instance = object.__new__(class_, *args, **kwargs)
    return class_._instance

  def __init__(self) -> None:
    self.__starts_model = joblib.load('SOS_model.pkl')
    self.__ends_model = joblib.load('EOS_model.pkl')
    self.__start_features = ['evi2', 'lswi', '2m_temp_max', 'skin_temp_max', 'gdd_2m_acc', 'daylength']
    self.__end_features = ['evi2', 'redcontrast', '2m_temp_max', 'skin_temp_max', 'total_evaporation', 'daylength']
    
  def predict(self, features: pd.Series[float | int] | pd.DataFrame, DOY: int | None = None) -> int | float | tuple[list[int | float], list[int]]:
    '''
    Use this method to predict the class label (0 for [**NO VEGETATION**] / 1 for [**VEGETATION IS PRESENT**] in this observation)
    DOY determines the model to be used & the features to be used.
    
    + **DOY >= 184 falls to the EOS (End Of Season) model**
    + **DOY < 184 falls to the SOS (Start Of Season) model**
    \\
    + The ***SOS*** model uses the following features: 
      - **EVI2**
      - **LSWI**
      - **2m_temperature_max** (in a 3-day rolling window)
      - **skin_temp_max** (in a 3-day rolling window)
      - **gdd_2m_acc** (cumulative sum of daily gdd)
      - **DayLength**
    + The ***EOS*** model uses the following features:
      - **EVI2**
      - **RedContrast**
      - **2m_temp_max** (in a 3-day rolling window)
      - **skin_temp_max** (in a 3-day rolling window)
      - **total_evaporation**
      - **DayLength**
    ---
    + ***!! NOTE THAT EACH FEATURE NAME MUST BE USED EXACTLY AS STATED IN THE DOCUMENTATION, EXCEPT FOR THE CASE !!***
    + ***To see how to calculate the features, refer to [AllFeatures.pdf] in the current directory.***
    '''
    
    EOS = True
    SOS = True
    prediction_sos = []
    prediction_eos = []
    X = features.copy(deep=True)
    
    if isinstance(X, pd.Series):
      X.index = [i.lower() for i in X.index]
      if DOY:
        if DOY >= 184:
          return self.__ends_model.predict(X[self.__end_features])
        else:
          return self.__starts_model.predict(X[self.__start_features])
      else:
        raise AssertionError('For a pd.Series object, DOY must be passed as an argument!')
      
    elif isinstance(X, pd.DataFrame):
      X.columns = [i.lower() for i in X.columns]
      X = X.sort_values(by='doy')
      
      sos = X[X['doy'] < 184]
      eos = X[X['doy'] >= 184]
      
      if not sos.empty:
        sos = sos[self.__start_features]
      else:
        print('Start-Of-Season data missing...')
        SOS = False
      
      if not eos.empty:
        eos = eos[self.__end_features]
      else:
        print('End-Of-Season data missing...')
        EOS = False
      
      if SOS:
        prediction_sos = self.__starts_model.predict(sos)
      
      if EOS:
        prediction_eos = self.__ends_model.predict(eos)
      
      return [*prediction_sos, *prediction_eos], X['doy'].tolist()
      
    else:
      raise AssertionError('Non-supported input type detected.')
