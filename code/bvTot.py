import io
import pyttsx3
import speech_recognition as sr


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def recog(filename):

    # with sr.Microphone() as source:
    #     print("Listening...")
    #     r.pause_threshold = 1
    #     audio = r.listen(source)

    r = sr.Recognizer()
    audio_data = sr.AudioFile(filename)
    with audio_data as source:
        audio_data = r.record(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio_data, language='bn-BD')
        # print(f"User said: {query}\n")
        with io.open("test.txt", "w", encoding="utf-8") as f:
            f.write(query)

        
        f = open("test.txt", "r", encoding="utf-8")
        text = f.readline()

        # Advance For Bangla
        text = query.strip().split(" ")
        sads = ["বিষণ্ণ","করুণ","বিমর্ষ","দুঃখিত","অসহায়","শক্তিহীন","অক্ষম","নিঃস্ব","বেচারা","কান্না"]
        happys = ["ফাটাফাটি","খুশী","সুখী","ভাগ্যবান","শোভন","হাসি","সুন্দর","আনন্দ","শুভ","আনন্দদায়ক"]
        angrys = ["ক্রুদ্ধ","ক্রোধ","হিংস্র","ক্ষ্যাপা","পাগলা","গালিগালাজ","রাগী","বদমেজাজ"]
        neutrals = ["কথা","সাধারণ","কথোপকথন","শিখতে"]

        # Sad section
        for sad in sads:
            for item in text:
                if item == sad:
                    # speak('I have your emotion sad.')
                    print('From Text: sad')
        # Happy Section
        for happy in happys:
            for item in text:
                if item == happy:
                    # speak('I have your emotion happy.')
                    print('From Test: happy')

        # Angry Section
        for angry in angrys:
            for item in text:
                if item == angry:
                    # speak('I have your emotion angry.')
                    print('From Text: angry')

        # Neutral Section
        for neutral in neutrals:
            for item in text:
                if item == neutral:
                    # speak('I have your emotion angry.')
                    print('From Text: neutral')
        # print('You have received Bangla text.')


    except Exception as e:
        print(e)
        # print("Say that again please...")