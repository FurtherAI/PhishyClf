import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras as k
from matplotlib import pyplot as plt
import ssl
import OpenSSL
import favicon
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import re


class SeqModel:
    def __init__(self, create_model=False):
        self.data = np.load(r'C:\Users\austi\PycharmProjects\PhishyClf\PhishData.npy', allow_pickle=True, fix_imports=False)
        self.data = self.data.astype(int)
        self.data[self.data == -1] = 0
        self.data = pd.DataFrame(self.data)
        self.data.drop([13, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29], axis='columns', inplace=True)
        self.x = np.array(self.data.iloc[:, :19])  # 2.2, 3.2, 3.4-3.5, 4.1-4.7, 1-12, 2-6, 3-5, 4-7
        self.y = np.array(self.data.iloc[:, -1])
        labels = []
        for i in self.y:
            if i == 1:
                labels.append([1, 0])
            else:
                labels.append([0, 1])
        self.y = np.array(labels)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=.2)

        if create_model:
            self.model = k.Sequential()
            self.model.add(k.layers.Embedding(input_dim=2, output_dim=32, input_length=19))
            self.model.add(k.layers.Flatten())
            self.model.add(k.layers.Dense(100, activation='relu', name='one'))
            self.model.add(k.layers.Dropout(.2))
            self.model.add(k.layers.Dense(100, activation='relu', name='two'))
            self.model.add(k.layers.Dropout(.2))
            self.model.add(k.layers.Dense(100, activation='relu', name='three'))
            self.model.add(k.layers.Dense(100, activation='relu', name='four'))
            self.model.add(k.layers.Dense(50, activation='relu', name='five'))
            self.model.add(k.layers.Dropout(.2))
            self.model.add(k.layers.Dense(2, activation='softmax', name='output'))
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[k.metrics.BinaryAccuracy(threshold=.5)])
            self.history = self.model.fit(self.x_train, self.y_train, batch_size=50, epochs=125, validation_data=(self.x_test, self.y_test))
            self.model.save(r'C:\Users\austi\PycharmProjects\PhishyClf\PhishModel')

        else:
            self.model = k.models.load_model(r'C:\Users\austi\PycharmProjects\PhishyClf\PhishModel')

    def predict(self, url):
        features = []

        # 1.1
        split_url = url.split('/')
        while '' in split_url:
            split_url.remove('')
        if 'http:' in split_url:
            split_url.remove('http:')
        if 'https:' in split_url:
            split_url.remove('https:')
        domain = split_url[0].replace('.', '')
        if 'x' in domain:
            try:
                test = [int(val, 16) for val in domain.split('x')]
                features.append(0)
            except ValueError:
                features.append(1)
        else:
            try:
                int(domain)
                features.append(0)
            except ValueError:
                features.append(1)

        # 1.2
        if len(url) >= 54: features.append(0)
        else: features.append(1)

        # 1.3
        if split_url[0] == 'bit.ly': features.append(0)
        else: features.append(1)

        # 1.4
        if '@' in url: features.append(0)
        else: features.append(1)

        # 1.5
        if url.rindex('//') > 7: features.append(0)
        else: features.append(1)

        # 1.6
        if '-' in split_url[0]: features.append(0)
        else: features.append(1)

        # 1.7
        domain = split_url[0].split('.')
        if 'www' in domain:
            domain.remove('www')
        if len(domain) > 2: features.append(0)
        else: features.append(1)

        # 1.8
        split_url = url.split('/')
        https = False
        while '' in split_url:
            split_url.remove('')
        if 'http:' in split_url:
            split_url.remove('http:')
        if 'https:' in split_url:
            https = True
            split_url.remove('https:')
        domain = split_url[0]
        cert = ssl.get_server_certificate((domain, 443))
        result = (OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, str.encode(cert)))
        if https:
            cert_start = (str(result.get_notBefore()))[2:10]
            age = (20 * 12 + 9.2) - ((int(cert_start[2:4]) * 12) + int(cert_start[4:6]) + int(cert_start[6:8]) / 30)
            if age >= 12: features.append(1)
            else: features.append(0)
        else: features.append(0)

        # 1.9
        cert_end = str(result.get_notAfter())[2:10]
        expiry = ((int(cert_end[2:4]) * 12) + int(cert_end[4:6]) + int(cert_end[6:8]) / 30) - (20 * 12 + 9.2)
        if expiry <= 12: features.append(0)
        else: features.append(1)

        # 1.10
        icon = favicon.get(url)
        if len(icon) > 0:
            icon = str(icon[0].url).split('/')
            while '' in icon:
                icon.remove('')
            if 'http:' in icon:
                icon.remove('http:')
            if 'https:' in icon:
                icon.remove('https:')
            if icon[0] in domain: features.append(1)
            else: features.append(0)

        # 1.11
        parse = urlparse(url)
        if parse.scheme == 'https' or parse.scheme == 'http': features.append(1)
        else: features.append(0)

        # 1.12
        if 'https' in domain: features.append(0)
        else: features.append(1)

        # 2.1
        response = requests.get(url)

        soup = BeautifulSoup(response.text, 'html.parser')

        whole = 0
        outward = 0

        imageTags = soup.findAll('img')

        for i in imageTags:
            if 'https://' in str(i) or 'http://' in str(i):
                if domain not in str(i):
                    outward += 1
            whole += 1

        videoTags = soup.findAll('video')
        for i in videoTags:
            if 'https://' in str(i) or 'http://' in str(i):
                if domain not in str(i):
                    outward += 1
            whole += 1

        soundTags = soup.findAll('audio')
        for i in soundTags:
            if 'https://' in str(i) or 'http://' in str(i):
                if domain not in str(i):
                    outward += 1
            whole += 1

        percent = outward / whole

        if percent < 61: features.append(1)
        else: features.append(0)

        # 2.3
        linksOutward = 0
        numTags = 0

        metaTags = soup.findAll('meta')
        for i in metaTags:
            if 'https://' in i or 'http://' in i:
                if domain not in i:
                    linksOutward += 1
            numTags += 1

        scriptTags = soup.findAll('script')
        for i in scriptTags:
            if 'https://' in i or 'http://' in i:
                if domain not in i:
                    linksOutward += 1
            numTags += 1

        linkTags = soup.findAll('link')
        for i in linkTags:
            if 'https://' in i or 'http://' in i:
                if domain not in i:
                    linksOutward += 1
            numTags += 1

        percent = linksOutward / numTags
        if percent < 45: features.append(1)
        else: features.append(0)

        # 2.4
        form = soup.findAll("form", action=re.compile(r"^about:blank"))
        form2 = soup.findAll('form', action=re.compile(r"^"""))
        if len(form) > 0 or len(form2) > 0: features.append(0)
        else: features.append(1)

        # 2.5
        mailto = soup.find_all("a", href=re.compile(r"^mailto:"))
        if len(mailto) > 0: features.append(0)
        else: features.append(1)

        # 2.6

        title = soup.title.string
        titleList = title.lower().split(' ')

        titles = [i for i in titleList if i in url.lower()]
        if len(titles) > 0: features.append(1)
        else: features.append(0)

        # 3.1
        request = requests.get(url).history
        if len(request) >= 2: features.append(0)
        else: features.append(1)

        # 3.3
        count = 0
        script = soup.findAll('script')

        for i in script:
            if 'event.button==2' in str(i):
                count = 1

        if count > 0: features.append(0)
        else: features.append(1)

        inp = np.array(features).reshape(1, 19)
        return self.model.predict(inp)

    def plot_(self):
        plt.plot(self.history.history['binary_accuracy'])
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('bce accuracy/loss')
        plt.ylabel('metrics')
        plt.xlabel('epoch')
        plt.legend(['b_acc', 'train_loss', 'val_loss'], loc='upper left')
        plt.show()


def main(url):
    model = SeqModel()
    pred = model.predict(url)
    if pred.argmax() == 0: return 1
    else: return 0

print(main('https://github.com/'))
