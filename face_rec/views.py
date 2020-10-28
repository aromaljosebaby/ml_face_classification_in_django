from django.shortcuts import render

from .models import UserImage

from django.conf import settings



import joblib




# Create your views here.




import os

import  json

from .functions import predicting_images_functions






def home(request):
    """Process images uploaded by users"""


    showing_default_upload__pic=True



    baseUrl = settings.BASE_DIR_ROOT + settings.STATIC_URL

    MEDIA_DIR=settings.MEDIA_ROOT
    context={}
    if request.method == 'POST':



        img = request.FILES.get('myfile')


        print(img,'jhhhhhhhhhhhhgccg')



        if img is not None:

            showing_default_upload__pic=False

            if_no_img_alert = False

            show_img=True
            if_no_img_uploaded=False

            sike = UserImage.objects.create(img=img)

            img_p=sike.img.url.split('/')[-1]


            img_path=os.path.join(MEDIA_DIR,img_p)


            model = joblib.load( baseUrl+'ML/saved_model.pkl')


            lol = predicting_images_functions(img_path)

            print(model.predict(lol))

            prediction = model.predict(lol)[0]
            #print(prediction,'*****************')

            f=open( baseUrl+'ML/class_dictionary.json')

            class_dict=json.load(f)

            #print(class_dict)

            prdection_name=class_dict[str(prediction)]


            context = {'sike': sike, 'prdection_name': prdection_name, 'show_img': show_img,
                       'if_no_img_uploaded': if_no_img_uploaded}

            return render(request, 'face_rec/trying.html', context=context)


        else:
            if_no_img_alert=True
            show_img = False


            context = { 'show_img': show_img,'if_no_img_alert':if_no_img_alert,'showing_default_upload__pic':showing_default_upload__pic}

        return render(request, 'face_rec/trying.html', context=context)

    else:

        context = {'showing_default_upload__pic':showing_default_upload__pic}

        return render(request, 'face_rec/trying.html',context=context)



def drag(request):

    return render(request,'face_rec/drag_n_drop.html', {})


