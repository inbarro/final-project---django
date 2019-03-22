from django.shortcuts import render
# from django.http import HttpResponse


def home(request):
    return render(request, 'searchengine/home.html', {'title': 'home'})


def about(request):
    return render(request, 'searchengine/about.html', {'title': 'About'})

