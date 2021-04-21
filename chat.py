import json
import random
import torch
from nltkt import bag_of, tokenized
from model import NeuralNet
import pyaudio
import speech_recognition as sr
import pyttsx3 



r=sr.Recognizer()
engine= pyttsx3.init()
def speak(audio):

    engine.say(audio)
    engine.runAndWait()

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

f= open("intent.json","r")
a=json.load(f)

FILE ="data.pth"
data=torch.load(FILE)

input_size  = data["input_size"]
hidden_size  = data["hidden_size"]
output_size  = data["output_size"]
allwords = data["allwords"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Mark"
print("lets chat! type 'quit ' to exit ")
while True:
    with sr.Microphone() as source:
        print("listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio=r.listen(source)
        sentence=r.recognize_google(audio, language='en-in')
        sentence=sentence.lower()
        print("you said:",sentence,"/n")
    if sentence == "quit":
        break
    sentence=tokenized(sentence)
    x=bag_of(sentence, allwords)
    x=x.reshape(1, x.shape[0])
    x=torch.from_numpy(x)

    output=model(x)
    _, predicted = torch.max(output, dim=1)
    tag= tags[predicted.item()]

    probs= torch.softmax(output, dim=1)
    prob= probs[0][predicted.item()]

    if prob.item() > 0.75 :
        for intent in a["intents"]:
            if tag == intent["tag"]:
                query=random.choice(intent['responses'])
                print (f"{bot_name}:{random.choice(intent['responses'])}")
                speak(query)
                    
    else:
        print(f"{bot_name}: I do not understand ...")
        speak("I do not understand")
            





