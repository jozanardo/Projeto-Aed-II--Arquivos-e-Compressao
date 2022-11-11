#importando o OpenCV
import cv2
from google.colab.patches import cv2_imshow

#importando bibliotecas úteis
import numpy as np
from scipy import fft
from collections import Counter

#captura o vídeo
cap = cv2.VideoCapture('https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4')

#verifica se é possível abrir o vídeo
if (cap.isOpened() == False):
  print("Erro ao abrir o arquivo.")

c = 0 #gambiarra pra pegar um só frame

#enquanto o vídeo estiver aberto
while(cap.isOpened() and c == 0):
  #faz a leitura do frame
  ret, frame = cap.read()

  #verifica se foi possível ler o frame
  if ret == True:
    #converte a imagem para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    c += 1
  else:
    break
#cap.release()
#cv2.destroyAllWindows()

#dimensão da imagem
dimension = gray.shape
print("Dimensão da imagem em pixels:", dimension)

#mostrando os dados da imagem colorida
"""print(frame)
cv2_imshow(frame)"""

print(gray[0:8])
#cv2_imshow(gray)

#discrete cosine transform
dct = fft.dct(gray)
print(dct)

#alterando as cores da imagem
"""test = gray.copy()
for i in range(dimension[0]//2):
  for j in range(dimension[1]//2):
    test[i][j] = 100
print(test)
cv2_imshow(test)"""

#encontrando a frequência usando dicionários
dicionario = dict(sum(map(Counter, gray), Counter()))
print(dicionario)

#ordenando por menor frequência
from collections import OrderedDict
dicionario = OrderedDict(sorted(dicionario.items(), key=lambda x: x[1]))

#percorrendo o dicionário
for chave, valor in dicionario.items():
    print(f"{chave}: {valor}")