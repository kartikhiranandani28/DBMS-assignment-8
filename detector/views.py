from django.shortcuts import render
import joblib

# Load the model and vectorizer
model = joblib.load('detector/spam_detector_model.joblib')
vectorizer = joblib.load('detector/vectorizer.joblib')

def index(request):
    result = None
    if request.method == 'POST':
        email_text = request.POST.get('email')
        email_vectorized = vectorizer.transform([email_text])
        prediction = model.predict(email_vectorized)
        result = 'Spam' if prediction[0] == 1 else 'Not Spam'
    return render(request, 'detector/index.html', {'result': result})
