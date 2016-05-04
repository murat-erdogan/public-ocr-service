from django.shortcuts import render
import requests
from django.views.decorators.csrf import csrf_exempt
from ocr_service import utils


@csrf_exempt
def binarize(request):
    content = {}
    if request.method == 'POST':
        response = requests.get(request.POST['url'])
        if response.status_code == 200:
            binarized = utils.binarize(response.content)
            content = {"image": binarized}
    return render(request, 'binarize.html', content)


@csrf_exempt
def face_detect(request):
    content = {}
    if request.method == 'POST':
        response = requests.get(request.POST['url'])
        if response.status_code == 200:
            detected = utils.face_detect(response.content)
            content = {"image": detected}
    return render(request, 'face_detect.html', content)


@csrf_exempt
def text_detect(request):
    content = {}
    if request.method == 'POST':
        response = requests.get(request.POST['url'])
        if response.status_code == 200:
            detected = utils.text_detect(response.content)
            content = {"image": detected}
    return render(request, 'text_detect.html', content)


def index(request):
    return render(request, 'index.html')