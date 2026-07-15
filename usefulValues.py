from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import pandas as pd

# No idea where this came from
# I'll figure out later
# there are others like this, but this one seems important
columns_to_use=['age', 'pre_systolic', 'pre_diastolic', 'pre_heart_rate', 'pre_temp', 
                'pre_total_chol', 'pre_triglic', 'pre_urea',
                'pre_creatin', 'pre_fasting_glucose', 'pre_TGO_AST', 'pre_TPG_ALT',
                'pre_alkaline_phophat', 'pre_albumin', 'pre_hb', 
                'pre_erythrocytes', 'pre_hematocrit', 'pre_VCM', 'pre_HCM', 'pre_CHCM', 
                'pre_leukocytes', 'pre_segmenteds', 'pre_lymphocytes', 'pre_monocytes', 'pre_platelets', 'pre_density_urine', 
                'pre_mucus_filaments_urine', 'pre_crystals_urine', 'pre_bacterial_floral_urine',
                'sex', 'BMI']

# Obtained from PFI based variable removal using SVR
columns_to_drop_SVR=['pre_erythrocytes', 'pre_creatin', 'pre_RDW', 'sex', 'pre_epithelial_cells_urine', 
                  'pre_total_chol', 'height', 'weight', 'pre_leukocytes_urine', 'pre_triglic', 'pre_urea', 
                  'pre_ketone_bodies_urine', 'pre_alkaline_phophat', 'pre_HCM', 'pre_ph_urine', 
                  'pre_lymphocytes', 'pre_erythrocytes_urine', 'pre_VCM', 'pre_leukocytes', 'pre_CHCM', 
                  'pre_segmenteds', 'age', 'BMI', 'pre_total_bilirrub', 'pre_density_urine', 'pre_total_prot', 
                  'pre_uric_acid', 'pre_direct_bilirrub', 'pre_TGO_AST',  
                  'pre_anti_hbc', 'pre_anti_hiv', 'pre_anti_hcv', 'pre_beta_hcg', 'pre_HbsAg']

# Same as above, but including age, sex and BMI
columns_to_drop_SVR2 = ['pre_erythrocytes', 'pre_creatin', 'pre_RDW', 'pre_epithelial_cells_urine', 
                  'pre_total_chol', 'height', 'weight', 'pre_leukocytes_urine', 'pre_triglic', 'pre_urea', 
                  'pre_ketone_bodies_urine', 'pre_alkaline_phophat', 'pre_HCM', 'pre_ph_urine', 
                  'pre_lymphocytes', 'pre_erythrocytes_urine', 'pre_VCM', 'pre_leukocytes', 'pre_CHCM', 
                  'pre_segmenteds', 'pre_total_bilirrub', 'pre_density_urine', 'pre_total_prot', 
                  'pre_uric_acid', 'pre_direct_bilirrub', 'pre_TGO_AST',  
                  'pre_anti_hbc', 'pre_anti_hiv', 'pre_anti_hcv', 'pre_beta_hcg', 'pre_HbsAg']

# Obtained from PFI based variable removal using MLP
columns_to_drop_MLP = ['pre_erythrocytes_urine', 'pre_creatin', 'pre_erythrocytes', 'height',
                       'pre_hematocrit', 'weight', 'pre_alkaline_phophat', 'pre_VCM', 'pre_HCM',
                       'pre_RDW', 'pre_lymphocytes', 'pre_direct_bilirrub', 'pre_TPG_ALT', 'pre_heart_rate', 
                       'pre_monocytes', 'pre_total_bilirrub', 'pre_diastolic', 'pre_leukocytes', 
                       'pre_leukocytes_urine', 'pre_total_prot', 'pre_CHCM', 'pre_ph_urine', 
                       'pre_systolic', 'pre_basophils', 'pre_crystals_urine', 'pre_uric_acid', 
                       'pre_ketone_bodies_urine', 
                       'pre_anti_hbc', 'pre_anti_hiv', 'pre_anti_hcv', 'pre_beta_hcg', 'pre_HbsAg']


# Currently the best models

SVRModel = SVR(kernel='linear', C=396.4450376953816, epsilon=0.0001)
MLPModel = MLPRegressor(hidden_layer_sizes=(99,), activation='relu', solver='lbfgs', max_iter=5000, alpha=75)
RFModel = RandomForestRegressor(n_estimators=150, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, max_samples=1.0)

featureDict = pd.DataFrame(data={
                    'en':   ['Patient' , 'Height', 'Weight', 'BMI', 'Age'  , 'Sex' , 'Systolic pressure', 'Diastolic pressure', 'Heart rate'         , 'Body temperature'    , 'Total cholesterol', 'Triglycerides', 'Urea'    , 'Creatinine' , 'Fasting glucose'    , 'Uric acid'    , 'TGO AST'    , 'TPG ALT'    , 'Total bilirubin'   , 'Direct bilirubin'   , 'Indirect bilirubin'   , 'Alkaline phosphatase', 'Total protein' , 'Albumin'    , 'Beta HCG'    , 'Hemoglobin' , 'Erythrocytes'    , 'Hematocrit'    , 'VCM'    , 'HCM'    , 'CHCM'    , 'RDW'    , 'Leukocytes'    , 'Myelocytes'    , 'Metamyelocytes'    , 'Rods'      , 'Segmented neutrophils', 'Eosinophils'    , 'Basophils'    , 'Lymphocytes'    , 'Monocytes'    , 'Platelets'    , 'Anti HBc'    , 'Anti HCV'    , 'Anti HIV'    , 'HbsAg'    , 'Urine density'     , 'Urine protein'    , 'Urine glucose'    , 'Urine ketone bodies'      , 'Urine bilirubin'     , 'Urine nitrite'    , 'Urine epithelial cells'     , 'Urine mucus filaments'      , 'Urine leukocytes'    , 'Urine erythrocytes'    , 'Urine cilinders'    , 'Urine crystals'    , 'Urine pH'    , 'Urine hemoglobin'    , 'Urine bacterial flora'     , 'Urine leukocyte esterase'    , 'Test AUC 0-t'],
                    'pt':   ['Paciente', 'Altura', 'Peso'  , 'IMC', 'Idade', 'Sexo', 'Pressão sistólica', 'Pressão diastólica', 'Frequência cardíaca', 'Temperatura corporal', 'Colesterol total' , 'Triglicérides', 'Ureia'   , 'Creatinina' , 'Glicose em jejum'   , 'Ácido úrico'  , 'TGO AST'    , 'TPG ALT'    , 'Bilirrubina total' , 'Bilirrubina direta' , 'Bilirrubina indireta'	, 'Fosfatase alcalina'  , 'Proteína total', 'Albumina'   , 'Beta HCG'    , 'Hemoglobina', 'Eritrócitos'     , 'Hematócrito'   , 'VCM'    , 'HCM'    , 'CHCM'    , 'RDW'    , 'Leucócitos'    , 'Mielócitos'    , 'Metamielócitos'    , 'Bastonetes', 'segmentados'          , 'Eosinófilos'    , 'Basófilos'    , 'Linfócitos'     , 'Monócitos'    , 'Plaquetas'    , 'Anti-HBc'    , 'Anti-HCV'    , 'Anti-HIV'    , 'HbsAg'    , 'Densidade da urina', 'Proteína na urina', 'Glicose na urina' , 'Corpos cetônicos na urina', 'Bilirrubina na urina', 'Nitrito na urina' , 'Células epiteliais na urina', 'Filamentos de muco na urina', 'Leucócitos na urina' , 'Hemácias na urina'     , 'Cilindros na urina' , 'Cristais na urina' , 'pH da urina' , 'Hemoglobina na urina', 'Flora bacteriana na urina' , 'Leucócito esterase na urina' , 'ASC 0-t de teste'],
                    'code': ['patient' , 'height', 'weight', 'BMI', 'age'  , 'sex' , 'pre_systolic'     , 'pre_diastolic'     , 'pre_heart_rate'     , 'pre_temp'            , 'pre_total_chol'   , 'pre_triglic'  , 'pre_urea', 'pre_creatin', 'pre_fasting_glucose', 'pre_uric_acid', 'pre_TGO_AST', 'pre_TPG_ALT', 'pre_total_bilirrub', 'pre_direct_bilirrub', 'pre_indirect_bilirrub', 'pre_alkaline_phophat', 'pre_total_prot', 'pre_albumin', 'pre_beta_hcg', 'pre_hb'     , 'pre_erythrocytes', 'pre_hematocrit', 'pre_VCM', 'pre_HCM', 'pre_CHCM', 'pre_RDW', 'pre_leukocytes', 'pre_myelocytes', 'pre_metamyelocytes', 'pre_rods'  , 'pre_segmenteds'       , 'pre_eosinophils', 'pre_basophils', 'pre_lymphocytes', 'pre_monocytes', 'pre_platelets', 'pre_anti_hbc', 'pre_anti_hcv', 'pre_anti_hiv', 'pre_HbsAg', 'pre_density_urine' , 'pre_prot_urine'   , 'pre_glucose_urine', 'pre_ketone_bodies_urine'  , 'pre_bilirubin_urine' , 'pre_nitrite_urine', 'pre_epithelial_cells_urine' , 'pre_mucus_filaments_urine'  , 'pre_leukocytes_urine', 'pre_erythrocytes_urine', 'pre_cilinders_urine', 'pre_crystals_urine', 'pre_ph_urine', 'pre_hb_urine'        , 'pre_bacterial_floral_urine', 'pre_leukocyte_esterase_urine', 'test_AUC_0_t']
                    },
             index=['patient', 'height', 'weight', 'BMI', 'age', 'sex', 'pre_systolic', 'pre_diastolic', 'pre_heart_rate', 'pre_temp', 'pre_total_chol', 'pre_triglic', 'pre_urea', 'pre_creatin', 'pre_fasting_glucose', 'pre_uric_acid', 'pre_TGO_AST', 'pre_TPG_ALT', 'pre_total_bilirrub', 'pre_direct_bilirrub', 'pre_indirect_bilirrub', 'pre_alkaline_phophat', 'pre_total_prot', 'pre_albumin', 'pre_beta_hcg', 'pre_hb', 'pre_erythrocytes', 'pre_hematocrit', 'pre_VCM', 'pre_HCM', 'pre_CHCM', 'pre_RDW', 'pre_leukocytes', 'pre_myelocytes', 'pre_metamyelocytes', 'pre_rods', 'pre_segmenteds', 'pre_eosinophils', 'pre_basophils', 'pre_lymphocytes', 'pre_monocytes', 'pre_platelets', 'pre_anti_hbc', 'pre_anti_hcv', 'pre_anti_hiv', 'pre_HbsAg', 'pre_density_urine', 'pre_prot_urine', 'pre_glucose_urine', 'pre_ketone_bodies_urine', 'pre_bilirubin_urine', 'pre_nitrite_urine', 'pre_epithelial_cells_urine', 'pre_mucus_filaments_urine', 'pre_leukocytes_urine', 'pre_erythrocytes_urine', 'pre_cilinders_urine', 'pre_crystals_urine', 'pre_ph_urine', 'pre_hb_urine', 'pre_bacterial_floral_urine', 'pre_leukocyte_esterase_urine', 'test_AUC_0_t']
             )