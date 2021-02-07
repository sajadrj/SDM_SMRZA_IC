photo_dir='C:/Users/Asus/ImageCaption_newEnv_captionApi/SDM_SMRZA_IC/Picture2.jpg' #photo_2021-02-06_19-01-15.jpg

def imagecaption(photo_dir):    
    from pickle import load
    from numpy import argmax
    from keras.preprocessing.sequence import pad_sequences
    from keras.applications.vgg16 import VGG16
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.models import Model
    from keras.models import load_model
    from pathlib import Path

    tokenizer_dir='C:/Users/Asus/ImageCaption_newEnv_captionApi/SDM_SMRZA_IC/tokenizer.pkl'
    model_dir='C:/Users/Asus/ImageCaption_newEnv_captionApi/SDM_SMRZA_IC/modell_0.h5'
    
    def extract_feature(filename):
        model = VGG16()
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        return feature
    def word_for_id(integer, tokenizer):
    	for word, index in tokenizer.word_index.items():
    		if index == integer:
    			return word
    	return None
    def generate_desc(model, tokenizer, photo, max_length):
        in_text = 'startseq'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            prediction = model.predict([photo,sequence], verbose=0)
            prediction = argmax(prediction)
            word = word_for_id(prediction, tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        return in_text
    tokenizer = load(open(tokenizer_dir, 'rb'))
    max_length = 34
    model = load_model(model_dir)
    photo = extract_feature(photo_dir)
    description = generate_desc(model, tokenizer, photo, max_length)
    def convert(lst): 
        return ' '.join(lst).split() 
    
    def reduce(description):
        lst =  [description] 
        ls=convert(lst)
        ls.pop(0)
        ls.pop(-1)
        return ls
    description=reduce(description)  
    description=' '.join(description)
    #print(description)
    #translate
    !pip install google_trans_new
    from google_trans_new import google_translator  
    translator = google_translator()  
    translate_text = translator.translate(description ,lang_tgt='fa')  
    #print(translate_text)
    
    #tts
#    !pip install gTTS
#    text_file = open("C:/Users/Asus/ImageCaption_newEnv_captionApi/SDM_SMRZA_IC/ForVoice.txt", "w")
#    text_file.write(description)
#    text_file.close()
#    !gtts-cli  --file C:/Users/Asus/ImageCaption_newEnv_captionApi/SDM_SMRZA_IC/ForVoice.txt --output C:/Users/Asus/ImageCaption_newEnv_captionApi/SDM_SMRZA_IC/ForVoice.mp3
    return description,translate_text

imagecaption(photo_dir)
