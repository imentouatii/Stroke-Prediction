# app.py
# ===========================
# -  Alerte-Cœur  📈
# - PIN Médecin = 0000 (verrouillage simple)
# - Langues (FR / EN / AR) + RTL arabe
# - Téléphone UNIQUE (E.164)
# - Prédiction (RandomForest pipeline) + SHAP
# - Plan d’action Vert/Orange/Rouge
# - RDV: 1 patient = 1 RDV actif (EN ATTENTE ou CONFIRMÉ futur)
# - Médecin: confirme PLUSIEURS RDV (checkbox) + ANNULATION (EN ATTENTE / CONFIRMÉ)
# - “SMS” simulé (log) + persistance SQLite
# - Agenda 7 jours (past/free/occupé) + pas de RDV dans le passé
# - PDF téléchargeable même après reset formulaire
# - Exports CSV + XLSX si openpyxl dispo
# ===========================

import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date, timedelta
from io import BytesIO
import re
import uuid
import os
import sqlite3
import json

# ==================================================
# CONFIG PAGE  ✅ FIX SIDEBAR
# ==================================================
APP_TITLE_DEFAULT = "   Alerte-Cœur ... 📈"
st.set_page_config(
    page_title=APP_TITLE_DEFAULT,
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# PIN MEDECIN (simple)
# ==================================================
DOCTOR_PIN = "0000"

if "doctor_authed" not in st.session_state:
    st.session_state.doctor_authed = False

# ==================================================
# LANGUES (I18N)
# ==================================================
LANGS = {
    "🇫🇷 Français": "fr",
    "🇬🇧 English": "en",
    "🇸🇦 العربية": "ar",
}

I18N = {
    "fr": {
        "app_title": "  Alerte-Cœur ... 📈",
        "subtitle": "Prédiction + explicabilité (SHAP) + plan d’action + RDV (agenda réel simulé). ⚠️ Outil pédagogique.",
        "lang": "🌐 Langue",
        "mode": "Mode",
        "mode_patient": "👤 Patient",
        "mode_doctor": "🩺 Médecin",
        "doctor_pin_title": "🔐 Accès médecin",
        "doctor_pin_help": "Saisis le PIN pour accéder au mode Médecin.",
        "doctor_pin_label": "PIN Médecin",
        "doctor_pin_btn": "Déverrouiller",
        "doctor_pin_wrong": "PIN incorrect.",
        "doctor_pin_ok": "Accès Médecin autorisé ✅",
        "doctor_lock_btn": "Verrouiller (déconnexion)",
        "tabs_result": "📌 Résultat",
        "tabs_shap": "🧩 Explication (SHAP)",
        "welcome": "Bienvenue",
        "welcome_hint": "⬅️ Remplis le formulaire à gauche puis clique sur **Évaluer le risque**.",
        "last_pdf": "📄 Dernier PDF disponible",
        "download_last_pdf": "Télécharger le dernier PDF",
        "errors_title": "Erreurs / champs obligatoires",
        "fix": "Corrige :",
        "profile": "🧾 Profil patient",
        "profile_caption": "Tous les champs sont obligatoires. Téléphone UNIQUE (un patient = un numéro).",
        "identity": "👤 Identité",
        "lastname": "Nom",
        "firstname": "Prénom",
        "country_code": "Indicatif pays",
        "rule": "Règle :",
        "manual_full": "Numéro complet (sans +)",
        "local_number": "Numéro (sans 0, sans espaces)",
        "general": "1) Général",
        "exams": "2) Examens (listes)",
        "symptoms": "3) Symptômes / ECG",
        "evaluate": "🧠 Évaluer le risque",
        "risk_prob": "Probabilité de maladie cardiaque",
        "main_factors": "Facteurs principaux (résumé)",
        "pedago": "⚠️ Résultat pédagogique. Seul un professionnel de santé peut poser un diagnostic.",
        "rdv_block": "📅 Prendre un RDV (calendrier + heures libres) — téléphone unique",
        "rdv_exists": "RDV actif déjà existant pour ce patient.",
        "rdv": "Rendez-vous",
        "status": "Statut",
        "choose_date": "Choisir une date (fenêtre autorisée)",
        "free_hours": "Heures libres",
        "no_slot": "Aucun créneau libre (ou futur) ce jour. Choisis une autre date.",
        "ask_rdv": "✅ Confirmer la demande de RDV",
        "slot_taken": "Ce créneau vient d’être pris. Choisis une autre heure/date.",
        "rdv_requested": "RDV demandé ✅",
        "pdf_card": "📄 Télécharger le compte rendu (PDF avec SHAP)",
        "download_pdf": "Télécharger en PDF",
        "pdf_kept": "Le PDF reste disponible même si le formulaire se réinitialise.",
        "sms_patient": "📩 SMS reçus (simulation)",
        "no_sms": "Aucun SMS pour le moment.",
        "shap_caption": "Contributions positives = augmentent le risque. Contributions négatives = réduisent le risque.",
        "simple_explain": "🧾 Explication simple",
        "risk_low": "🟢 Risque faible",
        "risk_mid": "🟠 Risque modéré",
        "risk_high": "🔴 Risque élevé",
        "xlsx_warn": "XLSX non généré (openpyxl manquant). CSV OK. pip install openpyxl si besoin.",
        # médecin
        "doc_tabs_triage": "🩺 Triage",
        "doc_tabs_rdv": "✅ RDV",
        "doc_tabs_agenda": "📅 Agenda",
        "doc_tabs_sms": "📩 SMS",
        "doc_tabs_stats": "📊 Stats",
        "doc_table": "🩺 Tableau Médecin — triage & priorités (auto) + filtres (cases à cocher)",
        "no_patient": "Aucun patient évalué. Passe en mode Patient.",
        "filter_risk": "Filtre risque",
        "filter_rdv": "Filtre RDV",
        "search": "Recherche (nom / tel)",
        "confirm_pending": "✅ Confirmer des RDV EN ATTENTE (cases à cocher)",
        "none_pending": "Aucun RDV en attente.",
        "pending_list": "📌 Liste RDV en attente",
        "confirm_selected": "✅ Confirmer les RDV sélectionnés",
        "cancel_selected": "❌ Annuler les RDV sélectionnés",
        "confirmed_ok": "Confirmés ✅",
        "cancelled_ok": "Annulés ✅",
        "ignored": "Ignorés",
        "agenda_7": "📅 Agenda (aperçu 7 jours) — créneaux passés marqués 'past'",
        "sms_sent": "📩 SMS envoyés (log)",
        "stats_title": "📊 Statistiques (lecture seule)",
        "patients": "Patients",
        "rdv_total": "RDV (total)",
        "rdv_pending": "RDV EN ATTENTE",
        "rdv_confirmed": "RDV CONFIRMÉS",
        "rdv_cancelled": "RDV ANNULÉS",
        "distribution": "Répartition",
        "avg_risk": "Risque moyen (dernier score)",
        "risk_filter_red": "🔴 Rouge",
        "risk_filter_orange": "🟠 Orange",
        "risk_filter_green": "🟢 Vert",
        "sms_confirm_msg": "✅ RDV CONFIRMÉ : {rdv_text} (Alerte-Cœur)",
        "sms_cancel_msg": "❌ RDV annulé : {rdv_text} (Alerte-Cœur)",
    },
    "en": {
        "app_title": "💓📈   Heart-Alert  📈💓",
        "subtitle": "Prediction + explainability (SHAP) + action plan + appointments (simulated agenda). ⚠️ Educational tool.",
        "lang": "🌐 Language",
        "mode": "Mode",
        "mode_patient": "👤 Patient",
        "mode_doctor": "🩺 Doctor",
        "doctor_pin_title": "🔐 Doctor access",
        "doctor_pin_help": "Enter the PIN to access Doctor mode.",
        "doctor_pin_label": "Doctor PIN",
        "doctor_pin_btn": "Unlock",
        "doctor_pin_wrong": "Wrong PIN.",
        "doctor_pin_ok": "Doctor access granted ✅",
        "doctor_lock_btn": "Lock (logout)",
        "tabs_result": "📌 Result",
        "tabs_shap": "🧩 Explanation (SHAP)",
        "welcome": "Welcome",
        "welcome_hint": "⬅️ Fill the form on the left then click **Assess risk**.",
        "last_pdf": "📄 Last PDF available",
        "download_last_pdf": "Download last PDF",
        "errors_title": "Errors / required fields",
        "fix": "Please fix:",
        "profile": "🧾 Patient profile",
        "profile_caption": "All fields are required. Phone UNIQUE (one patient = one number).",
        "identity": "👤 Identity",
        "lastname": "Last name",
        "firstname": "First name",
        "country_code": "Country code",
        "rule": "Rule:",
        "manual_full": "Full number (without +)",
        "local_number": "Number (no leading 0, no spaces)",
        "general": "1) General",
        "exams": "2) Tests (lists)",
        "symptoms": "3) Symptoms / ECG",
        "evaluate": "🧠 Assess risk",
        "risk_prob": "Heart disease probability",
        "main_factors": "Main factors (summary)",
        "pedago": "⚠️ Educational result. Only a healthcare professional can diagnose.",
        "rdv_block": "📅 Book an appointment (calendar + free hours) — unique phone",
        "rdv_exists": "An active appointment already exists for this patient.",
        "rdv": "Appointment",
        "status": "Status",
        "choose_date": "Choose a date (allowed window)",
        "free_hours": "Available hours",
        "no_slot": "No available (or future) slot on this day. Choose another date.",
        "ask_rdv": "✅ Request appointment",
        "slot_taken": "This slot was just taken. Choose another date/time.",
        "rdv_requested": "Appointment requested ✅",
        "pdf_card": "📄 Download report (PDF with SHAP)",
        "download_pdf": "Download PDF",
        "pdf_kept": "The PDF stays available even if the form resets.",
        "sms_patient": "📩 Received SMS (simulation)",
        "no_sms": "No SMS yet.",
        "shap_caption": "Positive contributions increase risk. Negative contributions reduce risk.",
        "simple_explain": "🧾 Simple explanation",
        "risk_low": "🟢 Low risk",
        "risk_mid": "🟠 Moderate risk",
        "risk_high": "🔴 High risk",
        "xlsx_warn": "XLSX not generated (openpyxl missing). CSV OK. Install: pip install openpyxl.",
        # doctor
        "doc_tabs_triage": "🩺 Triage",
        "doc_tabs_rdv": "✅ Appointments",
        "doc_tabs_agenda": "📅 Agenda",
        "doc_tabs_sms": "📩 SMS",
        "doc_tabs_stats": "📊 Stats",
        "doc_table": "🩺 Doctor dashboard — triage & priority + filters (checkboxes)",
        "no_patient": "No patient evaluated yet. Switch to Patient mode.",
        "filter_risk": "Risk filter",
        "filter_rdv": "Appointment filter",
        "search": "Search (name / phone)",
        "confirm_pending": "✅ Confirm PENDING appointments (checkboxes)",
        "none_pending": "No pending appointments.",
        "pending_list": "📌 Pending list",
        "confirm_selected": "✅ Confirm selected",
        "cancel_selected": "❌ Cancel selected",
        "confirmed_ok": "Confirmed ✅",
        "cancelled_ok": "Cancelled ✅",
        "ignored": "Skipped",
        "agenda_7": "📅 Agenda (7-day view) — past slots marked 'past'",
        "sms_sent": "📩 Sent SMS (log)",
        "stats_title": "📊 Statistics (read-only)",
        "patients": "Patients",
        "rdv_total": "Appointments (total)",
        "rdv_pending": "PENDING",
        "rdv_confirmed": "CONFIRMED",
        "rdv_cancelled": "CANCELLED",
        "distribution": "Distribution",
        "avg_risk": "Average risk (latest score)",
        "risk_filter_red": "🔴 Red",
        "risk_filter_orange": "🟠 Orange",
        "risk_filter_green": "🟢 Green",
        "sms_confirm_msg": "✅ APPOINTMENT CONFIRMED: {rdv_text} (Heart-Alert)",
        "sms_cancel_msg": "❌ APPOINTMENT CANCELLED: {rdv_text} (Heart-Alert)",
    },
    "ar": {
        "app_title": "💓📈   إنذار-القلب  📈💓",
        "subtitle": "تنبؤ + تفسير (SHAP) + خطة عمل + مواعيد (أجندة محاكاة). ⚠️ أداة تعليمية.",
        "lang": "🌐 اللغة",
        "mode": "الوضع",
        "mode_patient": "👤 المريض",
        "mode_doctor": "🩺 الطبيب",
        "doctor_pin_title": "🔐 دخول الطبيب",
        "doctor_pin_help": "أدخل الرقم السري للوصول إلى وضع الطبيب.",
        "doctor_pin_label": "PIN الطبيب",
        "doctor_pin_btn": "فتح",
        "doctor_pin_wrong": "الرقم السري غير صحيح.",
        "doctor_pin_ok": "تم السماح بدخول الطبيب ✅",
        "doctor_lock_btn": "قفل (تسجيل خروج)",
        "tabs_result": "📌 النتيجة",
        "tabs_shap": "🧩 الشرح (SHAP)",
        "welcome": "مرحباً",
        "welcome_hint": "⬅️ املأ الاستمارة على اليسار ثم اضغط **تقييم الخطر**.",
        "last_pdf": "📄 آخر ملف PDF متاح",
        "download_last_pdf": "تحميل آخر PDF",
        "errors_title": "أخطاء / حقول إجبارية",
        "fix": "يرجى التصحيح:",
        "profile": "🧾 ملف المريض",
        "profile_caption": "كل الحقول إجبارية. رقم الهاتف فريد (مريض واحد = رقم واحد).",
        "identity": "👤 الهوية",
        "lastname": "اللقب",
        "firstname": "الاسم",
        "country_code": "رمز الدولة",
        "rule": "القاعدة:",
        "manual_full": "الرقم الكامل (بدون +)",
        "local_number": "الرقم (بدون 0 وبدون فراغات)",
        "general": "1) عام",
        "exams": "2) الفحوصات (قوائم)",
        "symptoms": "3) الأعراض / تخطيط القلب",
        "evaluate": "🧠 تقييم الخطر",
        "risk_prob": "احتمال مرض القلب",
        "main_factors": "أهم العوامل (ملخص)",
        "pedago": "⚠️ نتيجة تعليمية. التشخيص النهائي عند طبيب مختص.",
        "rdv_block": "📅 حجز موعد (تقويم + أوقات متاحة) — رقم هاتف فريد",
        "rdv_exists": "يوجد موعد نشط لهذا المريض بالفعل.",
        "rdv": "الموعد",
        "status": "الحالة",
        "choose_date": "اختر تاريخاً (ضمن الفترة المسموح بها)",
        "free_hours": "الأوقات المتاحة",
        "no_slot": "لا توجد أوقات متاحة (أو مستقبلية) في هذا اليوم. اختر تاريخاً آخر.",
        "ask_rdv": "✅ تأكيد طلب الموعد",
        "slot_taken": "هذا الموعد تم حجزه للتو. اختر وقتاً/تاريخاً آخر.",
        "rdv_requested": "تم طلب الموعد ✅",
        "pdf_card": "📄 تحميل التقرير (PDF مع SHAP)",
        "download_pdf": "تحميل PDF",
        "pdf_kept": "يبقى ملف PDF متاحاً حتى بعد إعادة ضبط الاستمارة.",
        "sms_patient": "📩 الرسائل المستلمة (محاكاة)",
        "no_sms": "لا توجد رسائل حتى الآن.",
        "shap_caption": "المساهمات الإيجابية تزيد الخطر، والسلبية تقلله.",
        "simple_explain": "🧾 شرح مبسط",
        "risk_low": "🟢 خطر منخفض",
        "risk_mid": "🟠 خطر متوسط",
        "risk_high": "🔴 خطر مرتفع",
        "xlsx_warn": "لم يتم إنشاء XLSX (openpyxl غير متوفر). CSV جاهز. ثبت: pip install openpyxl.",
        # doctor
        "doc_tabs_triage": "🩺 فرز",
        "doc_tabs_rdv": "✅ المواعيد",
        "doc_tabs_agenda": "📅 الأجندة",
        "doc_tabs_sms": "📩 الرسائل",
        "doc_tabs_stats": "📊 إحصائيات",
        "doc_table": "🩺 لوحة الطبيب — فرز وأولوية + فلاتر (مربعات اختيار)",
        "no_patient": "لا يوجد مرضى بعد. انتقل إلى وضع المريض.",
        "filter_risk": "فلتر الخطر",
        "filter_rdv": "فلتر الموعد",
        "search": "بحث (اسم / هاتف)",
        "confirm_pending": "✅ تأكيد المواعيد قيد الانتظار (مربعات اختيار)",
        "none_pending": "لا توجد مواعيد قيد الانتظار.",
        "pending_list": "📌 قائمة الانتظار",
        "confirm_selected": "✅ تأكيد المحدد",
        "cancel_selected": "❌ إلغاء المحدد",
        "confirmed_ok": "تم التأكيد ✅",
        "cancelled_ok": "تم الإلغاء ✅",
        "ignored": "تم التخطي",
        "agenda_7": "📅 أجندة (7 أيام) — الأوقات الماضية تظهر 'past'",
        "sms_sent": "📩 سجل الرسائل المرسلة",
        "stats_title": "📊 إحصائيات (قراءة فقط)",
        "patients": "المرضى",
        "rdv_total": "إجمالي المواعيد",
        "rdv_pending": "قيد الانتظار",
        "rdv_confirmed": "مؤكد",
        "rdv_cancelled": "ملغى",
        "distribution": "التوزيع",
        "avg_risk": "متوسط الخطر (آخر تقييم)",
        "risk_filter_red": "🔴 أحمر",
        "risk_filter_orange": "🟠 برتقالي",
        "risk_filter_green": "🟢 أخضر",
        "sms_confirm_msg": "✅ تم تأكيد الموعد: {rdv_text} (إنذار-القلب)",
        "sms_cancel_msg": "❌ تم إلغاء الموعد: {rdv_text} (إنذار-القلب)",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "fr"


def t(key: str) -> str:
    return I18N.get(st.session_state.lang, I18N["fr"]).get(key, I18N["fr"].get(key, key))


# ==================================================
# STYLE  ✅ (fond harmonisé + pas de zone noire)
# ==================================================
st.markdown(
    """
<style>
:root{
  --bg1:#0B1020;
  --bg2:#0A1A2D;
  --txt:rgba(255,255,255,0.92);
  --brand1:#00E5FF;
  --brand2:#7C4DFF;
  --brand3:#00FF9D;
}

/* ✅ Fond global harmonisé (plus de noir au centre) */
.stApp{
  background:
    radial-gradient(1200px 700px at 12% 8%, rgba(124,77,255,0.28), transparent 60%),
    radial-gradient(1100px 650px at 88% 12%, rgba(0,229,255,0.24), transparent 60%),
    radial-gradient(1200px 700px at 50% 115%, rgba(0,255,157,0.16), transparent 62%),
    linear-gradient(180deg, #091022 0%, #091b2d 45%, #062a3a 100%) !important;
  color: var(--txt) !important;
}
main .block-container{ background: transparent !important; }
[data-testid="stAppViewContainer"]{ background: transparent !important; }
header, section, footer { background: transparent !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background:
    radial-gradient(450px 300px at 50% 0%, rgba(0,229,255,0.14), transparent 65%),
    radial-gradient(450px 300px at 50% 35%, rgba(124,77,255,0.14), transparent 70%),
    linear-gradient(180deg, rgba(7,12,24,0.95), rgba(8,14,26,0.88));
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.92); }

/* ✅ Forcer textes gris en blanc partout (captions + markdown + labels) */
.stCaption,
.stMarkdown p,
.stMarkdown span,
.stMarkdown small,
small,
label,
div[data-testid="stMarkdownContainer"] * {
  color: rgba(255,255,255,0.96) !important;
  opacity: 1 !important;
}

/* ✅ FIX IMPORTANT: forcer le texte de st.text() en blanc */
div[data-testid="stText"] *{
  color: rgba(255,255,255,0.96) !important;
  opacity: 1 !important;
}

/* Badges */
.badge {
  display:inline-block; padding:3px 10px; border-radius:999px;
  font-weight:900; font-size:0.85rem; border:1px solid rgba(255,255,255,0.15);
}
.badge-green {background: rgba(0,255,120,0.13);}
.badge-orange{background: rgba(255,150,0,0.13);}
.badge-red{background: rgba(255,60,60,0.13);}

/* Cards */
.card {
  background:
    linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03)),
    linear-gradient(135deg, rgba(0,229,255,0.10), rgba(124,77,255,0.10));
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 22px;
  padding: 18px;
  box-shadow: 0 18px 40px rgba(0,0,0,0.35), 0 1px 0 rgba(255,255,255,0.06) inset;
  margin-bottom: 14px;
  backdrop-filter: blur(10px);
}
.card-title {
  font-size: 1.05rem;
  font-weight: 900;
  margin-bottom: 10px;
  color: rgba(255,255,255,0.96);
  display:flex;
  align-items:center;
  gap:10px;
}
.card-title:before{
  content:"";
  width:10px;
  height:10px;
  border-radius:999px;
  background: linear-gradient(135deg, var(--brand1), var(--brand2));
  box-shadow: 0 0 0 3px rgba(255,255,255,0.06);
}
.note {
  font-size: 0.95rem;
  color: rgba(255,255,255,0.96);
  line-height: 1.55rem;
  padding: 10px 12px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  box-shadow: 0 10px 24px rgba(0,0,0,0.22);
}
hr, .stMarkdown hr {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.16), transparent) !important;
  opacity: 1 !important;
  margin: 0.75rem 0 !important;
}

/* Buttons */
.stButton button, div[data-testid="stFormSubmitButton"] button, .stDownloadButton button {
  border-radius: 16px !important;
  font-weight: 950 !important;
  padding: 0.70rem 1.05rem !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  color: rgba(255,255,255,0.98) !important;
  background: linear-gradient(135deg, rgba(0,229,255,0.35), rgba(124,77,255,0.35)) !important;
  box-shadow: 0 14px 26px rgba(0,0,0,0.30) !important;
}
.stButton button:hover, div[data-testid="stFormSubmitButton"] button:hover, .stDownloadButton button:hover {
  filter: brightness(1.10);
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox div[data-baseweb="select"] > div {
  background: rgba(6, 10, 22, 0.82) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 16px !important;
  color: rgba(255,255,255,0.98) !important;
}
.stTextInput input::placeholder,
.stDateInput input::placeholder,
.stNumberInput input::placeholder { color: rgba(255,255,255,0.65) !important; opacity: 1 !important; }
.stDateInput svg { fill: rgba(255,255,255,0.92) !important; }

/* Number input: enlever + / - */
div[data-testid="stNumberInput"] button { display: none !important; }
div[data-testid="stNumberInput"] input { padding-right: 12px !important; }

/* Dataframe */
[data-testid="stDataFrame"]{
  border-radius: 18px !important;
  overflow: hidden !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  box-shadow: 0 12px 26px rgba(0,0,0,0.26) !important;
}
[data-testid="stDataFrame"] *{ color: rgba(255,255,255,0.96) !important; opacity: 1 !important; }

/* Hide footer/menu only (✅ keep header visible for sidebar toggle) */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


def card(title: str):
    st.markdown(f"<div class='card'><div class='card-title'>{title}</div>", unsafe_allow_html=True)


def end_card():
    st.markdown("</div>", unsafe_allow_html=True)


# ==================================================
# FILES
# ==================================================
DATA_FILE = "heart.csv"

# ✅ IMPORTANT: modèle RF (pipeline preprocess + clf=RandomForestClassifier)
# Assure-toi que ton script d'entraînement sauvegarde ce fichier.
MODEL_FILE = "heart_guard_best.joblib"

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

PATIENTS_CSV = os.path.join(EXPORT_DIR, "patients.csv")
APPOINTMENTS_CSV = os.path.join(EXPORT_DIR, "appointments.csv")
SMS_CSV = os.path.join(EXPORT_DIR, "sms_log.csv")

PATIENTS_XLSX = os.path.join(EXPORT_DIR, "patients.xlsx")
APPOINTMENTS_XLSX = os.path.join(EXPORT_DIR, "appointments.xlsx")
SMS_XLSX = os.path.join(EXPORT_DIR, "sms_log.xlsx")

DB_FILE = os.path.join(EXPORT_DIR, "heart_guard.db")


# ==================================================
# SQLITE
# ==================================================
def db_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def db_init():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS patients (
        patient_key TEXT PRIMARY KEY,
        payload TEXT NOT NULL
    )"""
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS appointments (
        appt_id TEXT PRIMARY KEY,
        payload TEXT NOT NULL
    )"""
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS sms_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        source TEXT,
        patient_name TEXT,
        phone_e164 TEXT,
        message TEXT
    )"""
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sms_phone ON sms_log(phone_e164)")
    conn.commit()
    conn.close()


def db_upsert_patient(patient_key: str, patient_dict: dict):
    conn = db_conn()
    conn.execute(
        "INSERT INTO patients(patient_key, payload) VALUES(?, ?) "
        "ON CONFLICT(patient_key) DO UPDATE SET payload=excluded.payload",
        (patient_key, json.dumps(patient_dict, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def db_upsert_appointment(appt_id: str, appt_dict: dict):
    conn = db_conn()
    conn.execute(
        "INSERT INTO appointments(appt_id, payload) VALUES(?, ?) "
        "ON CONFLICT(appt_id) DO UPDATE SET payload=excluded.payload",
        (appt_id, json.dumps(appt_dict, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def db_insert_sms(row: dict):
    conn = db_conn()
    conn.execute(
        "INSERT INTO sms_log(timestamp, source, patient_name, phone_e164, message) VALUES(?,?,?,?,?)",
        (
            row["timestamp"],
            row.get("source", "SYSTEM"),
            row.get("patient_name", ""),
            row.get("phone_e164", ""),
            row.get("message", ""),
        ),
    )
    conn.commit()
    conn.close()


def load_state_from_db_into_session():
    db_init()
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("SELECT patient_key, payload FROM patients")
    for patient_key, payload in cur.fetchall():
        st.session_state.patients[patient_key] = json.loads(payload)

    cur.execute("SELECT appt_id, payload FROM appointments")
    for appt_id, payload in cur.fetchall():
        st.session_state.appointments[appt_id] = json.loads(payload)

    cur.execute("SELECT timestamp, source, patient_name, phone_e164, message FROM sms_log ORDER BY id ASC")
    st.session_state.sms_log = [
        {"timestamp": ts, "source": src, "patient_name": pn, "phone_e164": ph, "message": msg}
        for (ts, src, pn, ph, msg) in cur.fetchall()
    ]
    conn.close()

    st.session_state.phone_index = {}
    for k, p in st.session_state.patients.items():
        pe = p.get("phone_e164")
        if pe:
            st.session_state.phone_index[pe] = k


# ==================================================
# SESSION STATE INIT
# ==================================================
def init_state():
    if "form_version" not in st.session_state:
        st.session_state.form_version = 0
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    if "patients" not in st.session_state:
        st.session_state.patients = {}
    if "phone_index" not in st.session_state:
        st.session_state.phone_index = {}
    if "appointments" not in st.session_state:
        st.session_state.appointments = {}
    if "schedule" not in st.session_state:
        st.session_state.schedule = {}
    if "sms_log" not in st.session_state:
        st.session_state.sms_log = []

    if "last_pdf_bytes" not in st.session_state:
        st.session_state.last_pdf_bytes = None
    if "last_pdf_name" not in st.session_state:
        st.session_state.last_pdf_name = None


init_state()


def sk(name: str) -> str:
    return f"in_{name}_v{st.session_state.form_version}"


def idk(name: str) -> str:
    return f"id_{name}_v{st.session_state.form_version}"


# ==================================================
# LOAD MODEL + DATA
# ==================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


model = load_model()

df = pd.read_csv(DATA_FILE, sep=";")
TARGET = "HeartDisease"
FEATURES = [c for c in df.columns if c != TARGET]
cat_cols = df[FEATURES].select_dtypes(include=["object"]).columns.tolist()

# ==================================================
# LABELS + HELP
# ==================================================
LABELS = {
    "Age": "Âge (années)",
    "RestingBP": "Tension au repos (mmHg)",
    "Cholesterol": "Cholestérol (mg/dL)",
    "FastingBS": "Glycémie à jeun > 120 mg/dL",
    "MaxHR": "Fréquence max (bpm)",
    "Oldpeak": "Oldpeak (ECG effort)",
    "Sex": "Sexe",
    "ChestPainType": "Type douleur thoracique",
    "RestingECG": "ECG au repos",
    "ExerciseAngina": "Angine à l’effort",
    "ST_Slope": "Pente ST",
}

CHEST_PAIN_LONG = (
    "✅ **TA (Typical Angina)** : douleur typique liée à l’effort, oppression, peut irradier.\n"
    "✅ **ATA (Atypical Angina)** : douleur moins typique.\n"
    "✅ **NAP (Non-Anginal Pain)** : douleur non cardiaque probable.\n"
    "✅ **ASY (Asymptomatic)** : pas de douleur thoracique.\n\n"
    "💡 *Ce champ aide le modèle à reconnaître des patterns associés au risque.*"
)

HELP = {
    "Age": "Âge (entier 1 → 120).",
    "RestingBP": "Pression artérielle au repos (mmHg).",
    "Cholesterol": "Cholestérol total (mg/dL).",
    "FastingBS": "0 = Non, 1 = Oui (glycémie à jeun > 120 mg/dL).",
    "MaxHR": "Fréquence cardiaque max à l’effort (bpm).",
    "Oldpeak": "Dépression du segment ST à l’effort (ECG).",
    "Sex": "Sexe biologique.",
    "RestingECG": "Résultat ECG au repos (Normal / LVH / ST…).",
    "ExerciseAngina": "Angine déclenchée par l’effort (Y/N).",
    "ST_Slope": "Pente du segment ST à l’effort (Up / Flat / Down).",
    "ChestPainType": CHEST_PAIN_LONG,
}

EMPTY = "— Sélectionner —"


def cat_options(col):
    return sorted(df[col].dropna().unique().tolist())


CAT_OPTIONS = {
    "Sex": cat_options("Sex"),
    "ChestPainType": cat_options("ChestPainType"),
    "RestingECG": cat_options("RestingECG"),
    "ExerciseAngina": cat_options("ExerciseAngina"),
    "ST_Slope": cat_options("ST_Slope"),
}

BP_OPTS = list(range(80, 201))
CHOL_OPTS = list(range(100, 401))
MAXHR_OPTS = list(range(60, 221))
OLDPEAK_OPTS = [round(x, 1) for x in np.arange(0.0, 6.1, 0.1)]
FASTING_OPTS = [0, 1]

MAP_DAYS = {
    "Monday": "Lundi",
    "Tuesday": "Mardi",
    "Wednesday": "Mercredi",
    "Thursday": "Jeudi",
    "Friday": "Vendredi",
    "Saturday": "Samedi",
    "Sunday": "Dimanche",
}
HOURS = ["08:30", "09:30", "10:30", "14:00", "15:00", "16:30"]

# ==================================================
# COUNTRY RULES
# ==================================================
COUNTRY_RULES = {
    "France (+33)": {"cc": "+33", "min": 9, "max": 9, "hint": "Ex: 6XXXXXXXX (9 chiffres, sans 0)"},
    "Algérie (+213)": {"cc": "+213", "min": 9, "max": 9, "hint": "Ex: 5XXXXXXXX (9 chiffres)"},
    "Maroc (+212)": {"cc": "+212", "min": 9, "max": 9, "hint": "Ex: 6XXXXXXXX (9 chiffres)"},
    "Tunisie (+216)": {"cc": "+216", "min": 8, "max": 8, "hint": "Ex: 2XXXXXXX (8 chiffres)"},
    "USA/Canada (+1)": {"cc": "+1", "min": 10, "max": 10, "hint": "Ex: 2025550123 (10 chiffres)"},
    "Autre (manuel)": {"cc": "+", "min": 6, "max": 14, "hint": "Saisis le numéro complet sans espaces (E.164 sans +)"},
}


def only_digits(s: str) -> str:
    return re.sub(r"\D", "", (s or ""))


def build_e164(country_label: str, local_number_raw: str, manual_full_raw: str) -> str | None:
    rule = COUNTRY_RULES[country_label]
    if country_label == "Autre (manuel)":
        digits = only_digits(manual_full_raw)
        if not (rule["min"] <= len(digits) <= rule["max"]):
            return None
        return f"+{digits}"
    digits = only_digits(local_number_raw)
    if not (rule["min"] <= len(digits) <= rule["max"]):
        return None
    return f"{rule['cc']}{digits}"


def valid_e164(e164: str) -> bool:
    return bool(re.fullmatch(r"\+[1-9]\d{5,14}", (e164 or "").strip()))


# ==================================================
# SCHEDULE
# ==================================================
def schedule_key(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def ensure_schedule_range(days: int = 60):
    today = date.today()
    for i in range(days + 1):
        d = today + timedelta(days=i)
        dk = schedule_key(d)
        if dk not in st.session_state.schedule:
            st.session_state.schedule[dk] = {}
        for h in HOURS:
            if h not in st.session_state.schedule[dk]:
                st.session_state.schedule[dk][h] = None


def is_slot_free(d: date, hour: str) -> bool:
    ensure_schedule_range(60)
    dk = schedule_key(d)
    return st.session_state.schedule[dk].get(hour) is None


def book_slot(d: date, hour: str, appt_id: str):
    ensure_schedule_range(60)
    dk = schedule_key(d)
    st.session_state.schedule[dk][hour] = appt_id


def get_free_hours_for_date(d: date):
    ensure_schedule_range(60)
    dk = schedule_key(d)
    return [h for h in HOURS if st.session_state.schedule[dk].get(h) is None]


def appt_datetime(a: dict) -> datetime | None:
    try:
        d_str = a.get("date")
        h_str = a.get("hour")
        if not d_str or not h_str:
            return None
        return datetime.strptime(f"{d_str} {h_str}", "%Y-%m-%d %H:%M")
    except Exception:
        return None


def rebuild_schedule_from_confirmed_appointments():
    ensure_schedule_range(60)
    for dk in st.session_state.schedule:
        for h in HOURS:
            st.session_state.schedule[dk][h] = None

    now_dt = datetime.now()
    for appt_id, a in st.session_state.appointments.items():
        if a.get("status") == "CONFIRMÉ":
            dt = appt_datetime(a)
            if dt is None:
                continue
            if dt >= now_dt:
                d_str = a.get("date")
                h = a.get("hour")
                if d_str in st.session_state.schedule and h in st.session_state.schedule[d_str]:
                    st.session_state.schedule[d_str][h] = appt_id


ensure_schedule_range(60)

# ==================================================
# EXPORTS
# ==================================================
def save_tables_to_disk():
    patients_df = pd.DataFrame(list(st.session_state.patients.values()))
    appts_df = pd.DataFrame(list(st.session_state.appointments.values()))
    sms_df = pd.DataFrame(st.session_state.sms_log)

    patients_df.to_csv(PATIENTS_CSV, index=False, encoding="utf-8")
    appts_df.to_csv(APPOINTMENTS_CSV, index=False, encoding="utf-8")
    sms_df.to_csv(SMS_CSV, index=False, encoding="utf-8")

    try:
        import openpyxl  # noqa: F401

        patients_df.to_excel(PATIENTS_XLSX, index=False)
        appts_df.to_excel(APPOINTMENTS_XLSX, index=False)
        sms_df.to_excel(SMS_XLSX, index=False)
        return True, None
    except Exception as e:
        return False, str(e)


# ==================================================
# SMS SIMULATION
# ==================================================
def send_sms(phone_e164: str, patient_name: str, message: str, source: str = "SYSTEM"):
    row = {
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "source": source,
        "patient_name": patient_name,
        "phone_e164": phone_e164,
        "message": message,
    }
    st.session_state.sms_log.append(row)
    db_insert_sms(row)
    save_tables_to_disk()


# ==================================================
# HELPERS
# ==================================================
def mask_phone(e164: str) -> str:
    if not e164 or not e164.startswith("+"):
        return e164 or ""
    digits = re.sub(r"\D", "", e164)
    if len(digits) <= 6:
        return e164
    return f"+{digits[:2]}******{digits[-2:]}"


def severity_rank(icon: str) -> int:
    return {"🔴": 0, "🟠": 1, "🟢": 2}.get(icon, 9)


def priority_score(proba_pct: float, input_data: dict) -> float:
    base = proba_pct / 100.0
    bonus = 0.0
    if input_data.get("ExerciseAngina") == "Y":
        bonus += 0.10
    if input_data.get("ChestPainType") == "TA":
        bonus += 0.10
    if float(input_data.get("RestingBP", 0)) >= 140:
        bonus += 0.05
    if int(input_data.get("FastingBS", 0)) == 1:
        bonus += 0.05
    if float(input_data.get("Oldpeak", 0)) >= 2.0:
        bonus += 0.05
    return round(base + bonus, 3)


def humanize_feature(name: str) -> str:
    if "_" in name:
        left, right = name.split("_", 1)
        mapping = {
            "Sex": "Sexe",
            "ChestPainType": "Douleur thoracique",
            "RestingECG": "ECG repos",
            "ExerciseAngina": "Angine effort",
            "ST_Slope": "Pente ST",
        }
        if left in mapping:
            return f"{mapping[left]} = {right}"
    return name


def simple_patient_explain(kind: str, top_lines: list[str]) -> str:
    base_map = {
        "green": {
            "fr": "Votre score est plutôt rassurant. Cela ne remplace pas un avis médical, mais vous êtes plutôt dans une zone de prévention.",
            "en": "Your score looks reassuring. This does not replace medical advice, but you are closer to prevention.",
            "ar": "النتيجة مطمئنة نسبياً. هذا لا يغني عن استشارة الطبيب، لكنها أقرب للوقاية.",
        },
        "orange": {
            "fr": "Votre score est intermédiaire. Il est recommandé de vérifier avec un médecin et d’améliorer les facteurs modifiables.",
            "en": "Your score is intermediate. It is recommended to consult a doctor and improve modifiable factors.",
            "ar": "النتيجة متوسطة. يُنصح باستشارة طبيب وتحسين العوامل القابلة للتغيير.",
        },
        "red": {
            "fr": "Votre score est élevé. Il est conseillé de consulter rapidement. En cas de symptômes importants, appelez les urgences.",
            "en": "Your score is high. It is recommended to consult quickly. If severe symptoms appear, call emergency services.",
            "ar": "النتيجة مرتفعة. يُنصح بمراجعة الطبيب بسرعة. إذا ظهرت أعراض قوية اتصل بالإسعاف.",
        },
    }
    base = base_map.get(kind, {}).get(st.session_state.lang, "")
    factors = "\n".join([ln.replace("- ", "• ") for ln in top_lines[:3]])
    head = {
        "fr": "Facteurs qui ont le plus influencé le score :",
        "en": "Top factors influencing the score:",
        "ar": "أهم العوامل التي أثرت على النتيجة:",
    }.get(st.session_state.lang, "Facteurs principaux :")
    return f"{base}\n\n{head}\n{factors}"


# ==================================================
# SHAP + PDF
# ==================================================
def shap_png_bytes(explanation: shap.Explanation) -> bytes:
    fig = plt.figure(figsize=(10, 4))
    shap.plots.waterfall(explanation, show=False, max_display=10)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_pdf_bytes(text: str, shap_png_data: bytes | None) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 50, "Alerte-Cœur — Compte rendu patient")

    c.setFont("Helvetica", 11)
    y = height - 80
    for line in text.split("\n"):
        if y < 260:
            break
        c.drawString(40, y, line[:120])
        y -= 14

    if shap_png_data is not None:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, 245, "Explication (SHAP) :")

        img = ImageReader(BytesIO(shap_png_data))
        img_w = width - 80
        img_h = 185
        c.drawImage(img, 40, 45, width=img_w, height=img_h, preserveAspectRatio=True, anchor="sw")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


# ==================================================
# DB LOAD ONCE
# ==================================================
if "db_loaded" not in st.session_state:
    load_state_from_db_into_session()
    rebuild_schedule_from_confirmed_appointments()
    st.session_state.db_loaded = True


# ==================================================
# LANG SELECTOR (SIDEBAR)
# ==================================================
_lang_values = list(LANGS.values())
_lang_keys = list(LANGS.keys())
try:
    _idx = _lang_values.index(st.session_state.lang)
except ValueError:
    _idx = 0

lang_label = st.sidebar.selectbox(t("lang"), _lang_keys, index=_idx)
st.session_state.lang = LANGS[lang_label]

# RTL pour arabe
if st.session_state.lang == "ar":
    st.markdown(
        """
        <style>
        html, body, [class*="st-"], .stApp {
            direction: rtl !important;
            text-align: right !important;
            font-family: "Tahoma","Arial","Noto Naskh Arabic","Amiri",sans-serif !important;
        }
        input, textarea { direction: rtl !important; text-align: right !important; }
        [data-testid="stDataFrame"] * { direction: ltr !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ==================================================
# HEADER
# ==================================================
st.title(t("app_title"))
st.markdown(f"<div class='note'>{t('subtitle')}</div>", unsafe_allow_html=True)

# ==================================================
# MODE (SIDEBAR)
# ==================================================
mode_value = st.sidebar.radio(
    t("mode"),
    options=["patient", "doctor"],
    index=0,
    format_func=lambda v: t("mode_patient") if v == "patient" else t("mode_doctor"),
)

# ✅ PIN FLOW: si mode doctor mais pas authed -> écran PIN
if mode_value == "doctor" and not st.session_state.doctor_authed:
    card(t("doctor_pin_title"))
    st.write(t("doctor_pin_help"))
    pin = st.text_input(t("doctor_pin_label"), type="password", key="doctor_pin_input")
    if st.button(t("doctor_pin_btn"), key="doctor_pin_btn"):
        if pin == DOCTOR_PIN:
            st.session_state.doctor_authed = True
            st.success(t("doctor_pin_ok"))
            st.rerun()
        else:
            st.error(t("doctor_pin_wrong"))
    end_card()
    st.stop()

# bouton lock quand médecin connecté
if mode_value == "doctor" and st.session_state.doctor_authed:
    if st.sidebar.button(t("doctor_lock_btn"), key="doctor_lock_btn"):
        st.session_state.doctor_authed = False
        st.rerun()

# ==================================================
# RDV HELPERS (ACTIVE)
# ==================================================
def patient_has_active_appointment(pkey: str):
    now_dt = datetime.now()
    for a in st.session_state.appointments.values():
        if a.get("patient_key") == pkey and a.get("status") in ("EN ATTENTE", "CONFIRMÉ"):
            if a.get("status") == "CONFIRMÉ":
                dt = appt_datetime(a)
                if dt is not None and dt < now_dt:
                    continue
            return a
    return None


def make_rdv_text(d: date, hour: str) -> str:
    day_fr = MAP_DAYS[d.strftime("%A")]
    return f"{day_fr} {d.strftime('%d/%m/%Y')} à {hour}"


def create_pending_appointment(pkey: str, d: date, hour: str):
    appt_id = str(uuid.uuid4())
    rdv_text = make_rdv_text(d, hour)
    p = st.session_state.patients[pkey]
    appt = {
        "appt_id": appt_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datetime": f"{d.strftime('%Y-%m-%d')} {hour}",
        "date": d.strftime("%Y-%m-%d"),
        "hour": hour,
        "rdv_text": rdv_text,
        "status": "EN ATTENTE",
        "confirmed_at": "",
        "cancelled_at": "",
        "cancelled_by": "",
        "patient_key": pkey,
        "patient_uuid": p["patient_uuid"],
        "patient_name": p["patient_name"],
        "phone_e164": p["phone_e164"],
        "status_icon": p["status_icon"],
        "risk_pct": p["risk_pct"],
        "priority_score": p["priority_score"],
    }
    st.session_state.appointments[appt_id] = appt
    st.session_state.patients[pkey]["rdv_status"] = "EN ATTENTE"
    st.session_state.patients[pkey]["rdv_text"] = rdv_text
    db_upsert_appointment(appt_id, appt)
    db_upsert_patient(pkey, st.session_state.patients[pkey])
    return appt


def cancel_appointment(appt_id: str, cancelled_by: str = "MEDECIN") -> bool:
    if appt_id not in st.session_state.appointments:
        return False

    appt = st.session_state.appointments[appt_id]

    if appt.get("status") == "ANNULÉ":
        return False

    if appt.get("status") == "CONFIRMÉ":
        d = datetime.strptime(appt["date"], "%Y-%m-%d").date()
        h = appt["hour"]
        dk = schedule_key(d)
        if dk in st.session_state.schedule and h in st.session_state.schedule[dk]:
            if st.session_state.schedule[dk].get(h) == appt_id:
                st.session_state.schedule[dk][h] = None

    appt["status"] = "ANNULÉ"
    appt["cancelled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    appt["cancelled_by"] = cancelled_by

    pkey = appt["patient_key"]
    if pkey in st.session_state.patients:
        st.session_state.patients[pkey]["rdv_status"] = "ANNULÉ"
        st.session_state.patients[pkey]["rdv_text"] = "(annulé)"

    send_sms(
        phone_e164=appt["phone_e164"],
        patient_name=appt["patient_name"],
        message=t("sms_cancel_msg").format(rdv_text=appt["rdv_text"]),
        source="MEDECIN",
    )

    db_upsert_appointment(appt_id, appt)
    if pkey in st.session_state.patients:
        db_upsert_patient(pkey, st.session_state.patients[pkey])

    return True


# ==================================================
# MEDECIN
# ==================================================
if mode_value == "doctor":
    t_triage, t_rdv, t_agenda, t_sms, t_stats = st.tabs(
        [t("doc_tabs_triage"), t("doc_tabs_rdv"), t("doc_tabs_agenda"), t("doc_tabs_sms"), t("doc_tabs_stats")]
    )

    with t_triage:
        card(t("doc_table"))
        if not st.session_state.patients:
            st.info(t("no_patient"))
            end_card()
            st.stop()

        patients_df = pd.DataFrame(list(st.session_state.patients.values()))
        patients_df["sev_rank"] = patients_df["status_icon"].apply(severity_rank)
        patients_df["tel_masque"] = patients_df["phone_e164"].apply(mask_phone)

        colf1, colf2, colf3 = st.columns([1, 1, 1])

        with colf1:
            st.markdown(f"**{t('filter_risk')}**")
            c_r = st.checkbox(t("risk_filter_red"), value=True, key="flt_risk_red")
            c_o = st.checkbox(t("risk_filter_orange"), value=True, key="flt_risk_orange")
            c_g = st.checkbox(t("risk_filter_green"), value=True, key="flt_risk_green")
            flt_icons = []
            if c_r:
                flt_icons.append("🔴")
            if c_o:
                flt_icons.append("🟠")
            if c_g:
                flt_icons.append("🟢")

        with colf2:
            st.markdown(f"**{t('filter_rdv')}**")
            c_none = st.checkbox(
                "AUCUN" if st.session_state.lang == "fr" else ("NONE" if st.session_state.lang == "en" else "بدون"),
                value=True,
                key="flt_rdv_none",
            )
            c_pending = st.checkbox(
                "EN ATTENTE"
                if st.session_state.lang == "fr"
                else ("PENDING" if st.session_state.lang == "en" else "قيد الانتظار"),
                value=True,
                key="flt_rdv_pending",
            )
            c_confirm = st.checkbox(
                "CONFIRMÉ"
                if st.session_state.lang == "fr"
                else ("CONFIRMED" if st.session_state.lang == "en" else "مؤكد"),
                value=True,
                key="flt_rdv_confirm",
            )
            c_cancel = st.checkbox(
                "ANNULÉ"
                if st.session_state.lang == "fr"
                else ("CANCELLED" if st.session_state.lang == "en" else "ملغى"),
                value=True,
                key="flt_rdv_cancel",
            )
            flt_rdv = []
            if c_none:
                flt_rdv.append("AUCUN")
            if c_pending:
                flt_rdv.append("EN ATTENTE")
            if c_confirm:
                flt_rdv.append("CONFIRMÉ")
            if c_cancel:
                flt_rdv.append("ANNULÉ")

        with colf3:
            q = st.text_input(t("search"), value="")

        if not flt_icons:
            flt_icons = ["🔴", "🟠", "🟢"]
        if not flt_rdv:
            flt_rdv = ["AUCUN", "EN ATTENTE", "CONFIRMÉ", "ANNULÉ"]

        dfv = patients_df.copy()
        dfv = dfv[dfv["status_icon"].isin(flt_icons)]
        dfv = dfv[dfv["rdv_status"].isin(flt_rdv)]
        if q.strip():
            qn = q.strip().lower()
            dfv = dfv[
                dfv["patient_name"].astype(str).str.lower().str.contains(qn)
                | dfv["phone_e164"].astype(str).str.lower().str.contains(qn)
                | dfv["tel_masque"].astype(str).str.lower().str.contains(qn)
            ]

        dfv = dfv.sort_values(["sev_rank", "priority_score", "risk_pct"], ascending=[True, False, False])

        st.dataframe(
            dfv[
                [
                    "timestamp",
                    "patient_uuid",
                    "patient_name",
                    "tel_masque",
                    "status_icon",
                    "risk_pct",
                    "priority_score",
                    "rdv_status",
                    "rdv_text",
                ]
            ],
            use_container_width=True,
        )
        end_card()

    with t_rdv:
        st.markdown(f"### {t('confirm_pending')}")
        pending = [a for a in st.session_state.appointments.values() if a.get("status") == "EN ATTENTE"]
        if not pending:
            st.info(t("none_pending"))
        else:
            appts_df = pd.DataFrame(pending)
            appts_df["sev_rank"] = appts_df["status_icon"].apply(severity_rank)
            appts_df = appts_df.sort_values(["sev_rank", "priority_score", "datetime"], ascending=[True, False, True])

            selected_appt_ids: list[str] = []
            card(t("pending_list"))
            for _, r in appts_df.iterrows():
                line = (
                    f"{r['status_icon']}  **{r['patient_name']}**  |  "
                    f"{mask_phone(r['phone_e164'])}  |  "
                    f"{r['rdv_text']}  |  "
                    f"Priorité: **{r['priority_score']}**"
                )
                if st.checkbox(line, key=f"chk_pending_{r['appt_id']}"):
                    selected_appt_ids.append(r["appt_id"])
            end_card()

            colA, colB = st.columns([1, 1])

            with colA:
                if st.button(t("confirm_selected"), disabled=(len(selected_appt_ids) == 0), key="btn_confirm_selected"):
                    confirmed = 0
                    skipped = 0
                    now_dt = datetime.now()

                    for appt_id in selected_appt_ids:
                        a = st.session_state.appointments.get(appt_id)
                        if not a or a.get("status") != "EN ATTENTE":
                            skipped += 1
                            continue

                        dt = appt_datetime(a)
                        if dt is not None and dt < now_dt:
                            skipped += 1
                            continue

                        d = datetime.strptime(a["date"], "%Y-%m-%d").date()
                        h = a["hour"]

                        if not is_slot_free(d, h):
                            skipped += 1
                            continue

                        book_slot(d, h, appt_id)
                        st.session_state.appointments[appt_id]["status"] = "CONFIRMÉ"
                        st.session_state.appointments[appt_id]["confirmed_at"] = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )

                        pkey = a["patient_key"]
                        if pkey in st.session_state.patients:
                            st.session_state.patients[pkey]["rdv_status"] = "CONFIRMÉ"
                            st.session_state.patients[pkey]["rdv_text"] = a["rdv_text"]

                        send_sms(
                            phone_e164=a["phone_e164"],
                            patient_name=a["patient_name"],
                            message=t("sms_confirm_msg").format(rdv_text=a["rdv_text"]),
                            source="MEDECIN",
                        )

                        db_upsert_appointment(appt_id, st.session_state.appointments[appt_id])
                        if pkey in st.session_state.patients:
                            db_upsert_patient(pkey, st.session_state.patients[pkey])

                        confirmed += 1

                    rebuild_schedule_from_confirmed_appointments()
                    ok_xlsx, _ = save_tables_to_disk()

                    st.success(f"{t('confirmed_ok')} : {confirmed} | {t('ignored')}: {skipped}")
                    if not ok_xlsx:
                        st.warning(t("xlsx_warn"))
                    st.rerun()

            with colB:
                if st.button(t("cancel_selected"), disabled=(len(selected_appt_ids) == 0), key="btn_cancel_selected"):
                    cancelled = 0
                    skipped = 0
                    for appt_id in selected_appt_ids:
                        ok = cancel_appointment(appt_id, cancelled_by="MEDECIN")
                        if ok:
                            cancelled += 1
                        else:
                            skipped += 1

                    rebuild_schedule_from_confirmed_appointments()
                    ok_xlsx, _ = save_tables_to_disk()

                    st.success(f"{t('cancelled_ok')} : {cancelled} | {t('ignored')}: {skipped}")
                    if not ok_xlsx:
                        st.warning(t("xlsx_warn"))
                    st.rerun()

        st.markdown("---")
        st.subheader(
            "🗑️ "
            + (
                "Annuler des RDV CONFIRMÉS"
                if st.session_state.lang == "fr"
                else ("Cancel CONFIRMED appointments" if st.session_state.lang == "en" else "إلغاء المواعيد المؤكدة")
            )
        )

        now_dt = datetime.now()
        confirmed_future = []
        for a in st.session_state.appointments.values():
            if a.get("status") == "CONFIRMÉ":
                dt = appt_datetime(a)
                if dt is not None and dt >= now_dt:
                    confirmed_future.append(a)

        if not confirmed_future:
            st.info(
                "Aucun RDV confirmé futur."
                if st.session_state.lang == "fr"
                else ("No future confirmed appointments." if st.session_state.lang == "en" else "لا توجد مواعيد مؤكدة مستقبلية.")
            )
        else:
            cdf = pd.DataFrame(confirmed_future)
            cdf["sev_rank"] = cdf["status_icon"].apply(severity_rank)
            cdf = cdf.sort_values(["sev_rank", "priority_score", "datetime"], ascending=[True, False, True])

            selected_conf_ids: list[str] = []
            card(
                "📌 "
                + (
                    "RDV confirmés (futurs)"
                    if st.session_state.lang == "fr"
                    else ("Confirmed (future)" if st.session_state.lang == "en" else "المواعيد المؤكدة (مستقبلية)")
                )
            )
            for _, r in cdf.iterrows():
                line = (
                    f"{r['status_icon']}  **{r['patient_name']}**  |  "
                    f"{mask_phone(r['phone_e164'])}  |  "
                    f"{r['rdv_text']}  |  "
                    f"Priorité: **{r['priority_score']}**"
                )
                if st.checkbox(line, key=f"chk_conf_{r['appt_id']}"):
                    selected_conf_ids.append(r["appt_id"])
            end_card()

            if st.button(t("cancel_selected"), disabled=(len(selected_conf_ids) == 0), key="btn_cancel_confirmed"):
                cancelled = 0
                skipped = 0
                for appt_id in selected_conf_ids:
                    ok = cancel_appointment(appt_id, cancelled_by="MEDECIN")
                    if ok:
                        cancelled += 1
                    else:
                        skipped += 1

                rebuild_schedule_from_confirmed_appointments()
                ok_xlsx, _ = save_tables_to_disk()
                st.success(f"{t('cancelled_ok')} : {cancelled} | {t('ignored')}: {skipped}")
                if not ok_xlsx:
                    st.warning(t("xlsx_warn"))
                st.rerun()

    with t_agenda:
        card(t("agenda_7"))
        rows = []
        today = date.today()
        now_dt = datetime.now()
        for i in range(7):
            d = today + timedelta(days=i)
            dk = schedule_key(d)
            day_fr = MAP_DAYS[d.strftime("%A")]
            for h in HOURS:
                slot_dt = datetime.strptime(f"{dk} {h}", "%Y-%m-%d %H:%M")
                appt_id = st.session_state.schedule.get(dk, {}).get(h)

                if slot_dt < now_dt:
                    state = "past"
                else:
                    if appt_id is None:
                        state = "free"
                    else:
                        appt = st.session_state.appointments.get(appt_id)
                        state = appt["patient_name"] if appt else "occupied"

                rows.append({"date": d.strftime("%d/%m/%Y"), "jour": day_fr, "heure": h, "etat": state})

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        end_card()

    with t_sms:
        card(t("sms_sent"))
        if not st.session_state.sms_log:
            st.info(t("no_sms"))
        else:
            sdf = pd.DataFrame(st.session_state.sms_log)
            sdf["tel_masque"] = sdf["phone_e164"].apply(mask_phone)
            st.dataframe(
                sdf.iloc[::-1][["timestamp", "source", "patient_name", "tel_masque", "message"]],
                use_container_width=True,
            )
        end_card()

    with t_stats:
        card(t("stats_title"))
        patients_df = pd.DataFrame(list(st.session_state.patients.values()))
        appts_df = pd.DataFrame(list(st.session_state.appointments.values()))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(t("patients"), f"{len(patients_df)}")
        c2.metric(t("rdv_total"), f"{len(appts_df)}")
        if len(appts_df) > 0 and "status" in appts_df:
            c3.metric(t("rdv_pending"), f"{(appts_df['status']=='EN ATTENTE').sum()}")
            c4.metric(t("rdv_confirmed"), f"{(appts_df['status']=='CONFIRMÉ').sum()}")
            c5.metric(t("rdv_cancelled"), f"{(appts_df['status']=='ANNULÉ').sum()}")
        else:
            c3.metric(t("rdv_pending"), "0")
            c4.metric(t("rdv_confirmed"), "0")
            c5.metric(t("rdv_cancelled"), "0")

        if len(patients_df) > 0:
            dist = patients_df["status_icon"].value_counts().to_dict()
            st.write(f"{t('distribution')} : 🔴 {dist.get('🔴',0)} | 🟠 {dist.get('🟠',0)} | 🟢 {dist.get('🟢',0)}")
            st.write(f"{t('avg_risk')} : **{patients_df['risk_pct'].astype(float).mean():.1f}%**")
        end_card()

    st.stop()

# ==================================================
# PATIENT (SIDEBAR FORM)
# ==================================================
st.sidebar.header(t("profile"))
st.sidebar.caption(t("profile_caption"))

form_key = f"patient_form_v{st.session_state.form_version}"

with st.sidebar.form(form_key):
    st.subheader(t("identity"))
    last_name = st.text_input(t("lastname"), key=idk("last_name"))
    first_name = st.text_input(t("firstname"), key=idk("first_name"))

    country = st.selectbox(t("country_code"), list(COUNTRY_RULES.keys()), index=0, key=idk("country"))
    hint = COUNTRY_RULES[country]["hint"]
    st.caption(f"{t('rule')} {hint}")

    if country == "Autre (manuel)":
        manual_full = st.text_input(t("manual_full"), key=idk("manual_full"), placeholder="Ex: 33612345678")
        local_number = ""
    else:
        local_number = st.text_input(t("local_number"), key=idk("local_number"), placeholder="Ex: 612345678")
        manual_full = ""

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader(t("general"))

    with st.expander("ℹ️ Âge (années)"):
        st.write(HELP["Age"])
    age = st.number_input("Âge (entier)", min_value=1, max_value=120, step=1, format="%d", key=sk("Age"))

    with st.expander(f"ℹ️ {LABELS['Sex']}"):
        st.write(HELP["Sex"])
    sex = st.selectbox(LABELS["Sex"], [EMPTY] + CAT_OPTIONS["Sex"], key=sk("Sex"))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader(t("exams"))
    restingbp = st.selectbox(LABELS["RestingBP"], [EMPTY] + BP_OPTS, key=sk("RestingBP"))
    cholesterol = st.selectbox(LABELS["Cholesterol"], [EMPTY] + CHOL_OPTS, key=sk("Cholesterol"))
    fasting = st.selectbox(LABELS["FastingBS"], [EMPTY] + FASTING_OPTS, key=sk("FastingBS"))
    maxhr = st.selectbox(LABELS["MaxHR"], [EMPTY] + MAXHR_OPTS, key=sk("MaxHR"))
    oldpeak = st.selectbox(LABELS["Oldpeak"], [EMPTY] + OLDPEAK_OPTS, key=sk("Oldpeak"))

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader(t("symptoms"))
    with st.expander(f"ℹ️ {LABELS['ChestPainType']} (détails)"):
        st.markdown(CHEST_PAIN_LONG)
    chest = st.selectbox(LABELS["ChestPainType"], [EMPTY] + CAT_OPTIONS["ChestPainType"], key=sk("ChestPainType"))
    recg = st.selectbox(LABELS["RestingECG"], [EMPTY] + CAT_OPTIONS["RestingECG"], key=sk("RestingECG"))
    exang = st.selectbox(LABELS["ExerciseAngina"], [EMPTY] + CAT_OPTIONS["ExerciseAngina"], key=sk("ExerciseAngina"))
    st_slope = st.selectbox(LABELS["ST_Slope"], [EMPTY] + CAT_OPTIONS["ST_Slope"], key=sk("ST_Slope"))

    submitted = st.form_submit_button(t("evaluate"))

tab_result, tab_shap = st.tabs([t("tabs_result"), t("tabs_shap")])


def missing_any():
    errs = []
    if not last_name.strip():
        errs.append("Nom manquant" if st.session_state.lang == "fr" else ("Missing last name" if st.session_state.lang == "en" else "اللقب مفقود"))
    if not first_name.strip():
        errs.append("Prénom manquant" if st.session_state.lang == "fr" else ("Missing first name" if st.session_state.lang == "en" else "الاسم مفقود"))

    phone_e164 = build_e164(country, local_number, manual_full)
    if phone_e164 is None or not valid_e164(phone_e164):
        errs.append(
            "Téléphone invalide (règles pays / format)"
            if st.session_state.lang == "fr"
            else ("Invalid phone (rules/format)" if st.session_state.lang == "en" else "رقم هاتف غير صالح (القواعد/الصيغة)")
        )
    else:
        existing_key = st.session_state.phone_index.get(phone_e164)
        if existing_key is not None:
            if existing_key != f"{last_name.strip().lower()}|{first_name.strip().lower()}|{phone_e164}":
                errs.append(
                    "Téléphone déjà utilisé par un autre patient (numéro unique)."
                    if st.session_state.lang == "fr"
                    else ("Phone already used by another patient (unique number)." if st.session_state.lang == "en" else "رقم الهاتف مستخدم من قبل مريض آخر (رقم فريد).")
                )

    fields = {
        "Sex": sex,
        "RestingBP": restingbp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting,
        "MaxHR": maxhr,
        "Oldpeak": oldpeak,
        "ChestPainType": chest,
        "RestingECG": recg,
        "ExerciseAngina": exang,
        "ST_Slope": st_slope,
    }
    for k, v in fields.items():
        if v == EMPTY:
            errs.append(
                f"Champ manquant : {LABELS.get(k,k)}"
                if st.session_state.lang == "fr"
                else (f"Missing field: {LABELS.get(k,k)}" if st.session_state.lang == "en" else f"حقل ناقص: {LABELS.get(k,k)}")
            )

    return errs, phone_e164


if submitted:
    st.session_state.submitted = True

if not st.session_state.submitted:
    with tab_result:

        # ✅ Afficher le dernier PDF même après reset / RDV
        if st.session_state.last_pdf_bytes:
            card(t("last_pdf"))
            st.download_button(
                t("download_last_pdf"),
                data=st.session_state.last_pdf_bytes,
                file_name=st.session_state.last_pdf_name or "alerte_coeur_report.pdf",
                mime="application/pdf",
                key="download_last_pdf_btn",
            )
            end_card()

        card(t("welcome"))
        st.write(t("welcome_hint"))
        end_card()

    st.stop()

    st.stop()

errors, phone_e164 = missing_any()
if errors:
    with tab_result:
        card(t("errors_title"))
        st.error(t("fix"))
        for e in errors:
            st.write("•", e)
        end_card()
    st.stop()

# patient_key basé sur phone UNIQUE + identité
patient_key = f"{last_name.strip().lower()}|{first_name.strip().lower()}|{phone_e164}"
patient_name = f"{first_name.strip()} {last_name.strip()}".strip()
st.session_state.phone_index[phone_e164] = patient_key

# build X
input_data = {
    "Age": int(age),
    "Sex": sex,
    "RestingBP": float(restingbp),
    "Cholesterol": float(cholesterol),
    "FastingBS": int(fasting),
    "MaxHR": float(maxhr),
    "Oldpeak": float(oldpeak),
    "ChestPainType": chest,
    "RestingECG": recg,
    "ExerciseAngina": exang,
    "ST_Slope": st_slope,
}
X = pd.DataFrame([input_data], columns=FEATURES)

# predict
proba = float(model.predict_proba(X)[0][1])
proba_pct = proba * 100

if proba_pct >= 70:
    icon, label, kind = "🔴", "Risque élevé", "red"
    limit_days = 3
elif proba_pct >= 50:
    icon, label, kind = "🟠", "Risque modéré", "orange"
    limit_days = 30
else:
    icon, label, kind = "🟢", "Risque faible", "green"
    limit_days = 0

# SHAP (✅ robust pour éviter l'erreur (20,2) sur waterfall)
preprocess = model.named_steps["preprocess"]
clf = model.named_steps["clf"]
X_trans = preprocess.transform(X)

ohe = preprocess.named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(cat_cols) if cat_cols else []
feature_names = list(cat_names) + [c for c in FEATURES if c not in cat_cols]

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_trans)

# Normaliser en (n_samples, n_features) pour la classe 1
if isinstance(shap_values, list):
    sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
else:
    sv = shap_values
    # cas possible: (n_samples, n_features, 2)
    if hasattr(sv, "ndim") and sv.ndim == 3 and sv.shape[-1] >= 2:
        sv = sv[:, :, 1]

base = explainer.expected_value
if isinstance(base, (list, np.ndarray)) and len(np.atleast_1d(base)) > 1:
    base = np.atleast_1d(base)[1]

row_data = (
    X_trans[0].toarray().ravel()
    if hasattr(X_trans[0], "toarray")
    else np.array(X_trans[0]).ravel()
)

exp = shap.Explanation(values=sv[0], base_values=base, data=row_data, feature_names=feature_names)
shap_png = shap_png_bytes(exp)

abs_vals = np.abs(exp.values)
top_idx = np.argsort(abs_vals)[::-1][:3]
top_lines = []
for i in top_idx:
    fname = humanize_feature(exp.feature_names[i])
    direction = "augmente" if exp.values[i] > 0 else "réduit"
    top_lines.append(f"- {fname} : {direction} le risque (impact {exp.values[i]:.3f})")

# plan d’action
if kind == "green":
    action_title = "🟢 Plan d’action (Prévention)"
    action_text = (
        "• Continuer un mode de vie sain\n"
        "• Alimentation équilibrée + hydratation\n"
        "• Sport : 30 min, 3–5 fois/semaine\n"
        "• Sommeil / stress : améliorer si besoin\n"
        "• Contrôle annuel recommandé"
    )
elif kind == "orange":
    action_title = "🟠 Plan d’action (Risque modéré)"
    action_text = (
        "• Arrêter le tabac\n"
        "• Réduire / arrêter l’alcool\n"
        "• Renforcer le sport + régime alimentaire adapté\n"
        "• RDV spécialiste conseillé sous 1 mois"
    )
else:
    action_title = "🔴 Plan d’action (Urgence relative)"
    action_text = (
        "• RDV spécialiste conseillé dans les 3 jours\n"
        "• Si symptômes sévères → urgences\n"
        "• Éviter efforts intenses en attendant avis médical\n"
        "• Médecin dispo 7/7 (simulation)"
    )

prio = priority_score(proba_pct, input_data)

# upsert patient
if patient_key not in st.session_state.patients:
    patient_uuid = str(uuid.uuid4())[:8]
    st.session_state.patients[patient_key] = {
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "patient_uuid": patient_uuid,
        "patient_key": patient_key,
        "patient_name": patient_name,
        "phone_e164": phone_e164,
        "status_icon": icon,
        "risk_pct": round(proba_pct, 1),
        "priority_score": prio,
        "rdv_status": "AUCUN" if kind == "green" else "EN ATTENTE",
        "rdv_text": "(aucun)" if kind == "green" else "(à définir)",
    }
else:
    st.session_state.patients[patient_key].update(
        {
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "status_icon": icon,
            "risk_pct": round(proba_pct, 1),
            "priority_score": prio,
        }
    )

db_upsert_patient(patient_key, st.session_state.patients[patient_key])


def build_report_text(pkey: str) -> str:
    p = st.session_state.patients[pkey]
    if kind == "green":
        rdv_line = "Rendez-vous : (non nécessaire — prévention)"
    else:
        rdv_line = f"Rendez-vous : {p.get('rdv_text','(à définir)')} | Statut : {p.get('rdv_status','')}"
    return (
        "ALERTE-CŒUR — COMPTE RENDU PATIENT\n"
        f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
        f"Identité : {p['patient_name']}\n"
        f"Téléphone : {p['phone_e164']}\n\n"
        f"Risque estimé : {proba_pct:.1f}%\n"
        f"Statut : {label}\n"
        f"Indice de priorité : {p['priority_score']}\n"
        f"{rdv_line}\n\n"
        f"{action_title}\n{action_text}\n\n"
        "Facteurs principaux :\n" + "\n".join(top_lines) + "\n\n"
        "⚠️ Outil pédagogique — ne remplace pas un diagnostic médical."
    )


report_text = build_report_text(patient_key)
pdf_bytes = build_pdf_bytes(report_text, shap_png)
st.session_state.last_pdf_bytes = pdf_bytes
st.session_state.last_pdf_name = f"alerte_coeur_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

# ==================================================
# UI RESULT
# ==================================================
with tab_result:
    card(t("tabs_result"))
    st.metric(t("risk_prob"), f"{proba_pct:.1f}%")

    if kind == "green":
        st.markdown("<span class='badge badge-green'>🟢</span> " + t("risk_low"), unsafe_allow_html=True)
    elif kind == "orange":
        st.markdown("<span class='badge badge-orange'>🟠</span> " + t("risk_mid"), unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge badge-red'>🔴</span> " + t("risk_high"), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(action_title)

    # ✅ FIX: afficher le plan d'action en BLANC (plus de gris invisible)
    st.markdown(
        "<div style='color: rgba(255,255,255,0.96); font-size: 1.0rem; line-height: 1.65rem; white-space: pre-wrap;'>"
        + action_text.replace("\n", "<br>")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"### {t('main_factors')}")
    for line in top_lines:
        st.write(line)

    with st.expander(t("simple_explain")):
        st.write(simple_patient_explain(kind, top_lines))

    st.info(t("pedago"))
    end_card()

    if kind in ("orange", "red"):
        card(t("rdv_block"))
        active = patient_has_active_appointment(patient_key)
        if active:
            st.warning(
                f"{t('rdv_exists')}\n\n"
                f"• {t('rdv')} : **{active['rdv_text']}**\n"
                f"• {t('status')} : **{active['status']}**"
            )
        else:
            today = date.today()
            max_day = today + timedelta(days=limit_days)

            chosen_date = st.date_input(
                t("choose_date"),
                value=today,
                min_value=today,
                max_value=max_day,
                key="rdv_date_input",
            )

            free_hours = get_free_hours_for_date(chosen_date)
            now_dt = datetime.now()

            if chosen_date == date.today():
                free_hours = [
                    h
                    for h in free_hours
                    if datetime.strptime(f"{schedule_key(chosen_date)} {h}", "%Y-%m-%d %H:%M") >= now_dt
                ]

            if not free_hours:
                st.warning(t("no_slot"))
                chosen_hour = None
            else:
                chosen_hour = st.selectbox(t("free_hours"), free_hours, key="rdv_hour_input")

            if st.button(t("ask_rdv"), key="patient_confirm_rdv", disabled=(chosen_hour is None)):
                slot_dt = datetime.strptime(f"{schedule_key(chosen_date)} {chosen_hour}", "%Y-%m-%d %H:%M")
                if slot_dt < datetime.now():
                    st.error(t("no_slot"))
                elif not is_slot_free(chosen_date, chosen_hour):
                    st.error(t("slot_taken"))
                else:
                    appt = create_pending_appointment(patient_key, chosen_date, chosen_hour)
                    ok_xlsx, _ = save_tables_to_disk()
                    st.success(f"{t('rdv_requested')} : **{appt['rdv_text']}** (EN ATTENTE)")
                    if not ok_xlsx:
                        st.warning(t("xlsx_warn"))

                    st.session_state.form_version += 1
                    st.session_state.submitted = False
                    st.rerun()
        end_card()

    card(t("pdf_card"))
    st.download_button(
        t("download_pdf"),
        data=st.session_state.last_pdf_bytes,
        file_name=st.session_state.last_pdf_name or "alerte_coeur_report.pdf",
        mime="application/pdf",
    )
    st.caption(t("pdf_kept"))
    end_card()

    card(t("sms_patient"))
    sms_for_patient = [s for s in st.session_state.sms_log if s.get("phone_e164") == phone_e164]
    if not sms_for_patient:
        st.info(t("no_sms"))
    else:
        for s in sms_for_patient[::-1][:10]:
            src = s.get("source", "SYSTEM")
            st.write(f"**{s['timestamp']}** ({src}) — {s.get('message','')}")
    end_card()

with tab_shap:
    card(t("tabs_shap"))
    st.caption(t("shap_caption"))
    fig2 = plt.figure(figsize=(10, 4))
    shap.plots.waterfall(exp, show=False, max_display=10)
    st.pyplot(fig2, clear_figure=True)
    end_card()