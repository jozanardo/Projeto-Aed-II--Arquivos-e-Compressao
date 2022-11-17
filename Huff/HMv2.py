import cv2
import numpy as np
from copy import deepcopy as cp


class Node:
    def __init__(self, f, p, isLeaf=0):
        self.freq = f  # Armazena Intensidade
        self.prob = p  # Armazena probabilidade
        self.word = ""  # recebe o codigo binario do pixel
        # Ponteiros para os nodes filhos, C[0] para o filho da esquerda, c[1] para o filho da direita
        self.c = [None, None]
        self.isLeaf = isLeaf  # Flag para nodes filhos


class Image:
    def __init__(self):
        self.path_in = ""  # Local do arquivo de input
        self.path_out = ""  # Local do arquivo de output
        self.im = np.zeros(1)  # Imagem do input
        self.out = np.zeros(1)  # Imagem do output
        self.image_data = np.zeros(1)  # Lista de intensidades e dimensoes
        self.r = 0  # Linhas da imagem
        self.c = 0  # Colunas da imagem
        self.d = 0  # Profundidade / Canais da imagem

        # histograma, frequencia de cada simbolo
        self.hist = np.zeros(1)
        self.freqs = np.zeros(1)  # frequencias nao nulas

        # Dicionario de frequencias e probabilidades dos simbolos
        self.prob_dict = {}
        self.allNodes = []  # armazena todos os nós criados
        self.leafNodes = {}  # armazena os nós folhas
        self.root = Node(-1, -1)  # nó raiz com probabilidade 1

        # string codificada da imagem na forma binaria: "01001010101011......",
        # é interpretado da seguinte forma:  [r,c,d,[..pixels]]
        self.encodedString = ""

        # lista de inteiros decodificados ao ler o arquivo .bin gerado na codi-
        # ficacao, na forma: [456,342,3,34,2,120,44, ...... ], e é interpretado
        # como: [r,c,d,[..pxls]]
        self.decodeList = []

        # Binario do arquivo de inteiros [0,1,0,0,1,0,1,0,1,0,1,0,1,1,......]
        self.binaryFromFile = []

    # Checagem de codificacao

    def checkCoding(self):
        return np.all(self.im == self.out)

    # Leitura de imagem
    def readImage(self, path):
        self.path_in = path
        try:
            self.im = cv2.imread(path)
        except:
            print("Error in reading image")

    # Inicializador de objeto de imagem
    def initialise(self):
        self.r, self.c, self.d = self.im.shape

# Puxando valores r,c,d para a codificacao em image_data list
        temp_list = self.im.flatten()
        temp_list = np.append(temp_list, self.r)
        temp_list = np.append(temp_list, self.c)
        temp_list = np.append(temp_list, self.d)

        self.image_data = temp_list

# Criando histograma de intensidades vindo de image_data para criar frequencias
        self.hist = np.bincount(
            self.image_data, minlength=max(256, self.r, self.c, self.d))
        total = np.sum(self.hist)

# Extraindo frequencias nao nulas
        self.freqs = [i for i, e in enumerate(self.hist) if e != 0]
        self.freqs = np.array(self.freqs)

# Criando dicionario de probabilidades, cujas chaves sao os valores de
# intensidade e os dados sao os valores de probabilidade
        for i, e in enumerate(self.freqs):
            self.prob_dict[e] = self.hist[e]/total

# funcao de comparacao para ordenacao
    def outImage(self, path):
        self.path_out = path
        try:
            cv2.imwrite(self.path_out, self.out)
        except:
            print("Error in writing the image")

# Criando nodes de intensidade
    def buildNodes(self):
        for key in self.prob_dict:
            leaf = Node(key, self.prob_dict[key], 1)
            self.allNodes.append(leaf)

# funcao de comparacao para ordenacao
    def prob_key(self, e):
        return e.prob

# Criando arvore superior
    def upTree(self):

        import heapq
        self.buildNodes()         # Criando nodes

        # ordenando todos os nodes para criar a arvore superior
        workspace = sorted(cp(self.allNodes), key=self.prob_key)

        while 1:
            c1 = workspace[0]
            c2 = workspace[1]
            print(c2)
            workspace.pop(0)
            workspace.pop(0)

            # criando um novo node para as duas menores intensidades provaveis
            new_node = Node(-1, c1.prob+c2.prob)
            new_node.c[0] = c1
            new_node.c[1] = c2

            workspace = list(heapq.merge(
                workspace, [new_node], key=self.prob_key))  # Colocando os nodes criados em workspace
            # Interrompe se a probabilidade do node criado for 1, indicando que a arvore superior criada foi completada
            if (new_node.prob == 1.0):
                self.root = new_node  # armazena node de probabilidade 1 como raiz
                return

        # Criando a arvore inferior, ou seja, colocando os dados nos nodes folhas a partir da raiz
    def downTree(self, root, word):
        root.word = word
        if(root.isLeaf):
            self.leafNodes[root.freq] = root.word
        if(root.c[0] != None):
            self.downTree(root.c[0], word+'0')
        if(root.c[1] != None):
            self.downTree(root.c[1], word+'1')

    def huffmanAlgo(self):
        self.upTree()  # Cria arvore superior
        self.downTree(self.root, "")  # Cria arvore inferior

        dicti = {}  # armazenando dicionario de probabilidades em uma nova variavel
        # Evitando acessar "self." pois custa tempo, usar o dicionario criado eh mais rapido
        for key in self.leafNodes:
            dicti[key] = self.leafNodes[key]

        # armazenando self.encodedString em uma variavel encodedString
        # acessar self. custa muito tempo, ou seja, mesma coisa que no caso acima
        encodedString = ""
        encodedString += dicti[self.r]
        encodedString += dicti[self.c]
        encodedString += dicti[self.d]

        # Note que primeiro eh codificado as dimensoes, e depois eh codificado
        # cada pixel na terceira dimensao. Eh decodificado da mesma maneira
        for i in range(self.r):
            for j in range(self.c):
                for ch in range(self.d):
                    encodedString += dicti[self.im[i][j][ch]]

        self.encodedString = encodedString

    def sendBinaryData(self, path):
        # O self.encodedString eh apenas uma lista de strings com characteres char('0') & char('1')
        # Mas nao eh escrito diretamente char('0') and char('1'), pois cada um destes tem 1byte = 8bits, entao eh convertido char('0') e char('1') para binary(0) & binary(1)
        # Para isso eh utilizado a funcao bitstring da biblioteca BitArray
        from bitstring import BitArray
        file = open(path, 'wb')
        obj = BitArray(bin=self.encodedString)
        obj.tofile(file)
        file.close()

    def decode(self, path):
        # Apos ter o arquivo codificado, eh tentada a leitura deste
        # Para ler o arquivo binario (.bin) eh utilizado a biblioteca bitarray
        # (esta eh uma diferente da utilizada acima, BitArray)

        import bitarray
        self.binaryFromFile = bitarray.bitarray()
        with open(path, 'rb') as f:
            self.binaryFromFile.fromfile(f)

        # Tipo de dado de self.binaryFromFile eh uma lista de inteiros int(0) e int(1)

        decodeList = []
        root = self.root
        temp_root = cp(self.root)

        temp_r = 0
        temp_c = 0
        temp_d = 0

        # Igual na codificacao, o r,c,d eh primeiro retirado/decodificado

        for i, c_int in enumerate(self.binaryFromFile):
            if(temp_r != 0 and temp_c != 0 and temp_d != 0 and len(decodeList) == (temp_r*temp_c*temp_d + 3)):
                break
            if(temp_r == 0 and len(decodeList) >= 1):
                temp_r = decodeList[0]

            if(temp_c == 0 and len(decodeList) >= 2):
                temp_c = decodeList[1]

            if(temp_d == 0 and len(decodeList) >= 3):
                temp_d = decodeList[2]

            # Inicia a busca na raiz, se 0 vai pro node da esquerda, se 1 vai para node da direita, se for node folha, eh armazenado a intensidade na lista
            # e segue esse mesmo procedimento ate acher os valores de (r*c*d + 3), ie 3 dimensoes e intensidades r*c*d

            temp_root = temp_root.c[c_int]
            if(temp_root.isLeaf):
                decodeList.append(temp_root.freq)
                temp_root = root
                continue

        self.decodeList = decodeList

    def decodeIm(self, path):
        self.decode(path)

        # Extraindo valores das dimensoes de decodeList, ie os primeiros 3

        decodeList = self.decodeList
        out_r = decodeList[0]
        decodeList.pop(0)
        out_c = decodeList[0]
        decodeList.pop(0)
        out_d = decodeList[0]
        decodeList.pop(0)

        out = np.zeros((out_r, out_c, out_d))

        # Preenchendo a matriz de saida

        for i in range(len(decodeList)):
            id = i//out_d
            x = id//out_c
            y = id % out_c      # comparacao aritmetica basica para obtencao dos valores
            z = i % out_d
            out[x][y][z] = decodeList[i]
        out = out.astype(dtype=int)
        self.out = out

    def huffmanCode(self, input_pth, compressed_pth="./compressed.bin", output_pth="./output.png", toCheck=0):
        # Os passos abaixo sao para melhor visualizacao da execucao

        import time

        self.readImage(input_pth) #LEITURA DA IMG AQ
        self.initialise()
        print('Initialization Done\n')

        s = time.time()
        print('Coding Image started\n')
        self.huffmanAlgo()
        print('Coding Image completed\n')
        e = time.time()

        print("\nOriginal Size of Image : ", self.r*self.c*self.d*8, " bits")
        print("\nCompressed Size : ", len(self.encodedString), " bits")
        print("\nCompressed factor : ", self.r *
              self.c*self.d*8/(len(self.encodedString)), "\n")

        print("Took ", e-s, " sec to encode input image\n")
        print('Sending coded data\n')
        self.sendBinaryData(compressed_pth)
        print('Soded data sent\n')

        s = time.time()

        print('Started decoding compressed image\n')
        self.decodeIm(compressed_pth)
        self.outImage(output_pth)
        print('Completed decoding compressed image ( open output image from the above mentioned path) \n')

        e = time.time()
        print("Took ", e-s, " sec to decode compressed image\n")
        if(toCheck):
            print("Are both images same : ", self.checkCoding())

path = "teste.mp4"
vidObj = cv2.VideoCapture(path, apiPreference=cv2.CAP_MSMF)

count = 0     
success = 1

while success == 1:         
    success, imagens = vidObj.read()

    if success != False:
        im = Image()
        cv2.imwrite("Images/frame%d.jpg" % count, imagens)
        path = "Images/frame%d.jpg" % count
        framepath = "Images/frame%d.jpg" % count
        im.huffmanCode(path, "./compressed.bin", framepath, toCheck=1)
        count += 1     
          

