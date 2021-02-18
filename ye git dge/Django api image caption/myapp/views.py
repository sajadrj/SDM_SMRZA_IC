from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import Document
from .forms import DocumentForm


@csrf_exempt
def my_view(request):
    print(f"Great! You're using Python 3.6+. If you fail here, use the right version.")
    message = 'Upload as many files as you want!'
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(image=request.FILES['docfile'])
            newdoc.save()
            #f(voroodi) khorooji be return
            return HttpResponse(f'test: {newdoc.image.name}')
            #.url
        else:
            message = 'The form is not valid. Fix the following error:'
    else:
        form = DocumentForm()  # An empty, unbound form
    # Load documents for the list page
    documents = Document.objects.all()
    for d in documents:
        print(d.image)

    # Render list page with the documents and the form
    context = {'documents': documents, 'form': form, 'message': message}
    return render(request, 'list.html', context)
