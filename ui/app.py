import streamlit as st
import pandas as pd
import joblib
import os # هذا السطر مهم جداً

# 1. تحديد مكان ملف النموذج المحفوظ (المسار المضمون)
# هذا الكود يحدد المسار بناءً على موقع ملف app.py نفسه، ثم يعود للخلف ويجد models/final_model.pkl
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.pkl')

# 2. تحميل النموذج
try:
    model = joblib.load(MODEL_PATH)
# ... (بقية الكود)
except FileNotFoundError:
    st.error(f"Error: Model file not found at {MODEL_PATH}. Make sure you ran all notebook cells and the file is in the 'models' folder.")
    st.stop()
    
# 3. إعداد واجهة التطبيق
st.set_page_config(page_title="تنبؤ أمراض القلب", layout="centered")
st.title("🩺 تطبيق التنبؤ بأمراض القلب")
st.markdown("أدخل قيم المتغيرات الـ 13 التالية للمريض للحصول على تنبؤ.")

# ترتيب الأعمدة حسب نموذجك
features = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# حقول إدخال البيانات
with st.form("prediction_form"):
    
    # تقسيم الإدخالات إلى ثلاثة أعمدة لتنظيم الواجهة
    col1, col2, col3 = st.columns(3)
    
    # 📝 مدخلات العمود 1
    with col1:
        age = st.number_input('Age (العمر)', min_value=18, max_value=100, value=50)
        sex = st.selectbox('Sex (الجنس)', options=[1, 0], format_func=lambda x: 'ذكر (1)' if x == 1 else 'أنثى (0)')
        cp = st.selectbox('CP (نوع ألم الصدر)', options=[1, 2, 3, 4])
        trestbps = st.number_input('Trestbps (ضغط الدم أثناء الراحة)', min_value=90, max_value=200, value=120)

    # 📝 مدخلات العمود 2
    with col2:
        chol = st.number_input('Chol (الكوليسترول)', min_value=100, max_value=600, value=200)
        fbs = st.selectbox('FBS (> 120 mg/dl)', options=[0, 1], format_func=lambda x: 'نعم (1)' if x == 1 else 'لا (0)')
        restecg = st.selectbox('Restecg (نتائج تخطيط القلب)', options=[0, 1, 2])
        thalach = st.number_input('Thalach (أقصى معدل لضربات القلب)', min_value=70, max_value=220, value=150)
    
    # 📝 مدخلات العمود 3
    with col3:
        exang = st.selectbox('Exang (ذبحة صدرية ناجمة عن ممارسة الرياضة)', options=[0, 1], format_func=lambda x: 'نعم (1)' if x == 1 else 'لا (0)')
        oldpeak = st.number_input('Oldpeak (انخفاض قطعة ST)', min_value=0.0, max_value=6.2, value=1.0, step=0.1)
        slope = st.selectbox('Slope (ميل قمة ST)', options=[1, 2, 3])
        ca = st.number_input('CA (عدد الأوعية الملونة)', min_value=0.0, max_value=3.0, value=0.0, step=1.0)
        thal = st.selectbox('Thal (نتيجة الثاليوم)', options=[3.0, 6.0, 7.0])

    # زر التنبؤ
    submitted = st.form_submit_button("تنبؤ النتيجة")


# 4. منطق التنبؤ
if submitted:
    # إنشاء DataFrame بمدخلات المستخدم بنفس ترتيب أعمدة التدريب
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=features)
    
    # التنبؤ باستخدام النموذج
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("🎉 نتيجة التنبؤ:")
    
    if prediction == 1:
        st.error(f"⚠️ احتمالية عالية للإصابة بمرض القلب ({prediction_proba[1]*100:.2f}%)")
        st.markdown("*(التنبؤ يشير إلى أن المريض لديه تضييق في أكثر من 50% من الأوعية)*")
    else:
        st.success(f"✅ احتمالية منخفضة للإصابة بمرض القلب ({prediction_proba[0]*100:.2f}%)")
        st.markdown("*(التنبؤ يشير إلى أن المريض لديه تضييق في أقل من 50% من الأوعية)*")

        import streamlit as st
import pandas as pd
import joblib
import os # 💡 إضافة مكتبة os للتعامل مع المسارات

# 1. تحديد مكان ملف النموذج المحفوظ
# نستخدم os.path.join لضمان عمل المسار على جميع أنظمة التشغيل (Mac/Windows)
# المسار المطلق للملف سيكون: Heart_Disease_Project/models/final_model.pkl
# __file__ يمثل موقع app.py، ثم نعود خطوة للخلف ".." ونذهب إلى models/
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.pkl')

# 2. تحميل النموذج
try:
    # النموذج يحتوي على كل من المعالجة (Scaling/Encoding) والمصنف (Classifier)
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Error: Model file not found at {MODEL_PATH}. Make sure you ran all notebook cells and the file is in the 'models' folder.")
    st.stop()
# ... (بقية الكود تبقى كما هي)