import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle
from autogluon.tabular import TabularDataset, TabularPredictor
import shap
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.ensemble import RandomForestClassifier

# model = pickle.load(open('model.pkl', 'rb'))
# predictor=TabularPredictor.load('./autogluon_model/4h_ventonly')
predictor=TabularPredictor.load('./model/', require_py_version_match=False)
# encoder_dict = pickle.load(open('encoder.pkl', 'rb'))
# data=pd.read_pickle('./data/autogluon/4h/aki.pkl').iloc[:,:21]
data=pd.read_pickle('./aki.pkl').iloc[:,:21]
train_data, test_data = train_test_split(data, test_size=0.2, random_state=31)
X_train=train_data.drop(columns=['label'])
cols=['分钟二氧化碳产量', '分钟吸气潮气量', '动态顺应性', '吸入氧气浓度（监测）', '吸气峰值气道压力',
       '吸气时间（秒）', '吸气潮气量', '呼吸弱度指数', '呼吸机做功', '呼吸频率（监测）', '呼气分钟通气量',
       '呼气末二氧化碳分压', '呼气末二氧化碳浓度（%）', '呼气末正压', '呼气潮气量', '平均气道压力', '气压',
       '自主呼吸分钟通气量', '通气二氧化碳产量', '饱和度监测' ]   

def main(): 
    st.title("AKI Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">AKI Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    minute_co2 = st.text_input("分钟二氧化碳产量","213") 
    minute_tidal_vol = st.text_input("分钟吸气潮气量","9") 
    dynamic_adaptation = st.text_input("动态顺应性","64") 
    o2_concentration = st.text_input("吸入氧气浓度（监测）","45") 
    peak_airway_p = st.text_input("吸气峰值气道压力","18") 
    inspiration_time = st.text_input("吸气时间（秒）","1") 
    tidal_vol = st.text_input("吸气潮气量","492") 
    rwi = st.text_input("呼吸弱度指数","47") 
    vent_work = st.text_input("呼吸机做功","0.9") 
    rr = st.text_input("呼吸频率（监测）","18") 
    bmv = st.text_input("呼气分钟通气量","8") 
    ppco2 = st.text_input("呼气末二氧化碳分压","36") 
    ppco2_percent = st.text_input("呼气末二氧化碳浓度（%）","4") 
    peep = st.text_input("呼气末正压","7") 
    etd = st.text_input("呼气潮气量","488") 
    average_airway_p = st.text_input("平均气道压力","10") 
    pneumatic = st.text_input("气压","1000") 
    minute_ventilatory_ventilation = st.text_input("自主呼吸分钟通气量","5") 
    ventilated_co2 = st.text_input("通气二氧化碳产量","12") 
    saturation = st.text_input("饱和度监测","97") 


    if st.button("Predict"): 
        features = [[minute_co2,minute_tidal_vol,dynamic_adaptation,o2_concentration,peak_airway_p,inspiration_time,tidal_vol,rwi,vent_work,rr,
                     bmv,ppco2,ppco2_percent,peep,etd,average_airway_p,pneumatic,minute_ventilatory_ventilation,ventilated_co2,saturation]]
        data = {'分钟二氧化碳产量': minute_co2, '分钟吸气潮气量': minute_tidal_vol, '动态顺应性': dynamic_adaptation, '吸入氧气浓度（监测）': o2_concentration, '吸气峰值气道压力': peak_airway_p, 
                '吸气时间（秒）': inspiration_time, '吸气潮气量': tidal_vol, '呼吸弱度指数': rwi, '呼吸机做功': vent_work, '呼吸频率（监测）': int(rr), '呼气分钟通气量': bmv, '呼气末二氧化碳分压': ppco2,
                '呼气末二氧化碳浓度（%）':ppco2_percent,'呼气末正压':peep,'呼气潮气量':etd,'平均气道压力':average_airway_p,'气压':pneumatic,'自主呼吸分钟通气量':minute_ventilatory_ventilation,
                '通气二氧化碳产量':ventilated_co2,'饱和度监测':saturation}
        print(data)
        df=pd.DataFrame([list(data.values())], columns=[ '分钟二氧化碳产量', '分钟吸气潮气量', '动态顺应性', '吸入氧气浓度（监测）', '吸气峰值气道压力',
       '吸气时间（秒）', '吸气潮气量', '呼吸弱度指数', '呼吸机做功', '呼吸频率（监测）', '呼气分钟通气量',
       '呼气末二氧化碳分压', '呼气末二氧化碳浓度（%）', '呼气末正压', '呼气潮气量', '平均气道压力', '气压',
       '自主呼吸分钟通气量', '通气二氧化碳产量', '饱和度监测'  ])
    
        prediction = predictor.predict(df)
    
        output = int(prediction[0])
        if output == 1:
            text = "AKI"
        else:
            text = "NO AKI"

        st.success('Outcome is {}'.format(text))
    
        target_class = 1
        class AutogluonWrapper:
            def __init__(self, predictor, feature_names, target_class=None):
                self.ag_model = predictor
                self.feature_names = feature_names
                self.target_class = target_class
                if target_class is None and predictor.problem_type != 'regression':
                    print("Since target_class not specified, SHAP will explain predictions for each class")
            
            def predict_proba(self, X):
                if isinstance(X, pd.Series):
                    X = X.values.reshape(1,-1)
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X, columns=self.feature_names)
                preds = self.ag_model.predict_proba(X)
                if predictor.problem_type == "regression" or self.target_class is None:
                    return preds
                else:
                    return preds[self.target_class]
        negative_class = 0
        baseline = train_data[train_data.label==negative_class].drop(columns=['label']).sample(50, random_state=0)
        ag_wrapper = AutogluonWrapper(predictor, X_train.columns, target_class)
        explainer = shap.KernelExplainer(ag_wrapper.predict_proba, baseline)
        ROW_INDEX = 0  # index of an example datapoint
        # single_datapoint = X_train.iloc[[ROW_INDEX]]
        # single_prediction = ag_wrapper.predict_proba(df)

        shap_values_single = explainer.shap_values(df, nsamples=100)
        fig=shap.force_plot(explainer.expected_value, shap_values_single, df,matplotlib=True)
        st.pyplot(fig)
if __name__=='__main__': 
    main()
