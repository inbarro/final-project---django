from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm
from algorithms import TFIDF


def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Your account has been created! You are now able to log in')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': form})

@login_required
def engine(request):
    return render(request, 'users/engine.html')


def tfifd (request):
    data = TFIDF.cosine_similarity(10, "Falkland petroleum exploration")
    return render(request, 'users/tfidf.html', {'data': data})


def jaccard (request):
    return render(request, 'users/jaccard.html')


