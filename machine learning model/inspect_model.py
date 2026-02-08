import joblib

model = joblib.load("heart_guard_best.joblib")

print("TYPE DU MODELE :", type(model))
print()

# 1️⃣ Vérifier si c’est un Pipeline
if hasattr(model, "named_steps"):
    print("✅ C'est un Pipeline sklearn")
    print("Étapes du pipeline :")
    for name, step in model.named_steps.items():
        print(f" - {name} → {type(step)}")
else:
    print("❌ Ce n'est PAS un Pipeline sklearn")
