# importing libraries
import numpy as np
import math
import glob
import cv2, sys, os
from google.colab.patches import cv2_imshow
from scipy.fft import dct, idct
from collections import Counter
from google.colab import drive
drive.mount('/drive')

class RunLengthEncoding:
  def __init__(self, video_path):
    self.video_path = video_path

    # luminance quantization matrix
    self.lqm = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                         [12, 12, 14, 19, 26, 58, 60, 55],
                         [14, 13, 16, 24, 40, 57, 69, 56],
                         [14, 17, 22, 29, 51, 87, 80, 62],
                         [18, 22, 37, 56, 68, 109, 103, 77],
                         [24, 35, 55, 64, 81, 104, 113, 92],
                         [49, 64, 78, 87, 103, 121, 120, 101],
                         [72, 92, 95, 98, 112, 100, 103, 99]], np.int16)
        
    # chrominance quantization matrix
    self.cqm = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                         [18, 21, 29, 66, 99, 99, 99, 99],
                         [24, 26, 56, 99, 99, 99, 99, 99],
                         [47, 66, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99]], np.int16)

    # dimension (resized image)
    self.width = 800
    self.length = 1200

    # blocks size (8 x 8)
    self.block = 8

  # frame storage
  def im_save(self, frame, path, name, format):
    if not os.path.exists(path):
      os.makedirs(path)

    frame = frame if (format == "jpeg") else np.uint8(frame)
    return cv2.imwrite(f'{path}/{name}.{format}', frame)

  # frame txt storage
  def im_save_txt(self, bitstream, path, name):
    if not os.path.exists(path):
      os.makedirs(path)

    file1 = open(f'{path}/{name}.txt', "w")
    file1.write(bitstream)
    file1.close()

  # create a zero matrix
  def zero_matrix(self, l, w):
    return np.zeros([l, w], np.int16)

  # chroma subsampling
  def chroma_subsamplig(self, frame):
    # converting to YCrCb color space
    YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    #extrating chanels
    Y, Cr, Cb = cv2.split(YCrCb)

    # converting the pixel color range to [-128, 127] (8-bits image)
    Y = Y.astype(np.int16) - 128
    Cr = Cr.astype(np.int16) - 128
    Cb = Cb.astype(np.int16) - 128

    return Y, Cr, Cb

  # apply the discrete cosine transformation (DCT)
  def dct_2D(self, m):
    return dct(dct(m.T, norm='ortho').T, norm='ortho')

  # apply the quantization
  def quantization(self, m, mq):
    return np.round(m/mq)

  # catch 8 by 8 blocks and apply the DCT and quantization
  def dct_quantization(self, Y, Cr, Cb):
    length = self.length
    width = self.width
    block = self.block

    y = self.zero_matrix(length, width)
    cr = self.zero_matrix(length,width)
    cb = self.zero_matrix(length, width)

    for i in range(length//block):
      coli = i*block
      colf = (i+1)*block

      for j in range(width//block):
        lini = j*block
        linf = (j+1)*block

        # applying the DCT and quantization
        y[coli:colf, lini:linf] = self.quantization(self.dct_2D(Y[coli:colf, lini:linf]), self.lqm)
        cr[coli:colf, lini:linf] = self.quantization(self.dct_2D(Cr[coli:colf, lini:linf]), self.cqm)
        cb[coli:colf, lini:linf] = self.quantization(self.dct_2D(Cb[coli:colf, lini:linf]), self.cqm)

    return y, cr, cb

  # zigzag scan of a matrix
  def zigzag(self, input):
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]
    
    i = 0

    output = np.zeros((vmax*hmax), np.int16)

    while ((v < vmax) and (h < hmax)):
      if ((h + v) % 2) == 0:
        if (v == vmin):
          output[i] = input[v, h]

          if (h == hmax):
            v = v + 1
          else:
            h = h + 1                        
          i = i + 1

        elif ((h == hmax -1 ) and (v < vmax)):
          output[i] = input[v, h] 
          v = v + 1
          i = i + 1

        elif ((v > vmin) and (h < hmax -1 )):
          output[i] = input[v, h] 
          v = v - 1
          h = h + 1
          i = i + 1

      else:
        if ((v == vmax -1) and (h <= hmax -1)):
          output[i] = input[v, h] 
          h = h + 1
          i = i + 1
      
        elif (h == hmin):
          output[i] = input[v, h] 

          if (v == vmax -1):
            h = h + 1
          else:
            v = v + 1
          i = i + 1

        elif ((v < vmax -1) and (h > hmin)):
          output[i] = input[v, h] 
          v = v + 1
          h = h - 1
          i = i + 1

      if ((v == vmax-1) and (h == hmax-1)): 	
        output[i] = input[v, h] 
        break

    return output

  # inverse zigzag scan of a matrix
  def izigzag(self, input, vmax, hmax):
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax), np.int16)

    i = 0

    while ((v < vmax) and (h < hmax)):  	
      if ((h + v) % 2) == 0:
        if (v == vmin):
          output[v, h] = input[i]

          if (h == hmax):
            v = v + 1
          else:
            h = h + 1                        
          i = i + 1

        elif ((h == hmax -1 ) and (v < vmax)):
          output[v, h] = input[i] 
          v = v + 1
          i = i + 1

        elif ((v > vmin) and (h < hmax -1 )):
          output[v, h] = input[i] 
          v = v - 1
          h = h + 1
          i = i + 1

      else:
        if ((v == vmax -1) and (h <= hmax -1)):
          output[v, h] = input[i] 
          h = h + 1
          i = i + 1
          
        elif (h == hmin):
          output[v, h] = input[i] 
          if (v == vmax -1):
            h = h + 1
          else:
            v = v + 1
          i = i + 1
                          
        elif((v < vmax -1) and (h > hmin)):
          output[v, h] = input[i] 
          v = v + 1
          h = h - 1
          i = i + 1

      if ((v == vmax-1) and (h == hmax-1)):
        output[v, h] = input[i] 
        break

    return output
  
  # 8 x 8 blocks handling to apply zig zag
  def blocks_handling(self, chanel, nbh, nbw):
    for i in range(nbh):
      # compute start and end row index of the block
      row_ind_1 = i*self.block                
      row_ind_2 = row_ind_1+self.block    
      
      for j in range(nbw):
        # compute start & end column index of the block
        col_ind_1 = j*self.block                     
        col_ind_2 = col_ind_1+self.block
                    
        block = chanel[row_ind_1:row_ind_2, col_ind_1:col_ind_2]
        
        # reorder DCT coefficients in zig zag order by calling zig zag function
        # it will give a one dimentional array
        reordered = self.zigzag(block)

        # reshape the reorderd array back to (block size by block size) (here: 8-by-8)
        reshaped= np.reshape(reordered, (self.block, self.block))
        
        # copy reshaped matrix into padded_img on current block corresponding indices
        chanel[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = reshaped 

    return chanel

  # RLE algorithm
  def get_run_length_encoding(self, image):
    i = 0
    skip = 0
    stream = []    
    bitstream = ""
    image = image.astype(int)

    while i < image.shape[0]:
        if image[i] != 0:            
            stream.append((image[i],skip))
            bitstream = bitstream + str(image[i])+ " " +str(skip)+ " "
            skip = 0
        else:
            skip = skip + 1
        i = i + 1

    return bitstream

  # frame encoding
  def encode(self, frame):
    # resizing the frame (800 x 1200)
    frame = cv2.resize(frame, (self.width, self.length))

    # extracting chroma subsampling
    Y, Cr, Cb = self.chroma_subsamplig(frame)

    # applying the DCT and quantization techniques
    Y, Cr, Cb = self.dct_quantization(Y, Cr, Cb)

    # walks into frame of 8 by 8 blocks
    h = self.length
    w = self.width

    h = np.float32(h)
    w = np.float32(w)

    nbh = math.ceil(h/self.block)
    nbh = np.int32(nbh)

    nbw = math.ceil(w/self.block)
    nbw = np.int32(nbw)

    i = 0
    for chanel in [Y, Cr, Cb]:
      padded_img = self.blocks_handling(chanel, nbh, nbw)
      arranged = padded_img.flatten()

      # RLE encoded
      bitstream = self.get_run_length_encoding(arranged)
      bitstream = str(padded_img.shape[0]) + " " + str(padded_img.shape[1]) + " " + bitstream + ";"

      if (i == 0):
        txt_name = "luminance"
      elif (i == 1):
        txt_name = "chrominance-red"
      else:
        txt_name = "chrominance-blue"

      self.im_save_txt(bitstream, "Txt", txt_name)
      i = i + 1
  
  # apply the inverse quantization
  def iquantization(self, m, mq):
    return m * mq
  
  # apply the inverse discrete cosine transform
  def idct_2D(self, m):
    return idct(idct(m.T, norm='ortho').T, norm='ortho')

  # catch 8 by 8 blocks and apply the idct and iquantization
  def idct_quantization(self, Y, Cr, Cb):
    iy = self.zero_matrix(self.length, self.width)
    icr = self.zero_matrix(self.length, self.width)
    icb = self.zero_matrix(self.length, self.width)

    for i in range(self.length//self.block):
      coli = i*self.block
      colf = (i+1)*self.block
      for j in range(self.width//self.block):
        lini = j*self.block
        linf = (j+1)*self.block

        # applying the inverse quantization and DCT
        iy[coli:colf, lini:linf] = self.idct_2D(self.iquantization(Y[coli:colf, lini:linf], self.lqm))
        icr[coli:colf, lini:linf] = self.idct_2D(self.iquantization(Cr[coli:colf, lini:linf], self.cqm))
        icb[coli:colf, lini:linf] = self.idct_2D(self.iquantization(Cb[coli:colf, lini:linf], self.cqm))

    return iy, icr, icb

  # join luminance and chrominances into a image
  def YCrCb2RGB(self, Y, Cr, Cb):
    # resetting the pixels color interval [0, 255] and data type
    y = (Y + 128).astype(np.uint8)
    cr = (Cr + 128).astype(np.uint8)
    cb = (Cb + 128).astype(np.uint8)

    # merging channels
    YCrCb = cv2.merge((y, cr, cb))

    return cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)

  # frame decoding
  def decode(self, dimension, c):
    i = 0
    for chanel in ["luminance", "chrominance-red", "chrominance-blue"]:
      # Reading .txt to decode it as image
      with open(f'Txt/{chanel}.txt', 'r') as my_file:
        image = my_file.read()
      
      # splits into tokens seperated by space characters
      details = image.split()

      # just python-crap to get integer from tokens : h and w are height and width of image (first two items)
      h = int(''.join(filter(str.isdigit, details[0])))
      w = int(''.join(filter(str.isdigit, details[1])))

      # declare an array of zeros (It helps to reconstruct bigger array on which IDCT and all has to be applied)
      array = np.zeros(h*w).astype(int)

      # some loop var initialisation
      k = 0
      i = 2
      x = 0
      j = 0

      # This loop gives us reconstructed array of size of image
      while k < array.shape[0]:
        if(details[i] == ';'):
          break
        
        if "-" not in details[i]:
            array[k] = int(''.join(filter(str.isdigit, details[i])))        
        else:
            array[k] = -1*int(''.join(filter(str.isdigit, details[i])))        

        if(i+3 < len(details)):
            j = int(''.join(filter(str.isdigit, details[i+3])))

        if j == 0:
            k = k + 1
        else:                
            k = k + j + 1        

        i = i + 2

      array = np.reshape(array,(h,w))

      # loop for constructing intensity matrix form frequency matrix (IDCT and all)
      i = 0
      j = 0
      k = 0

      # initialisation of compressed image
      padded_img = np.zeros((h,w), np.int16)

      while i < h:
          j = 0
          while j < w:        
              temp_stream = array[i:i+8,j:j+8]
              block = self.izigzag(temp_stream.flatten(), int(self.block),int(self.block))
              padded_img[i:i+8,j:j+8] = block
              j = j + 8        
          i = i + 8

      if (chanel == "luminance"):
        Y = padded_img
      elif (chanel == "chrominance-red"):
        Cr = padded_img
      else:
        Cb = padded_img

    Y, Cr, Cb = self.idct_quantization(Y, Cr, Cb)
    frame = self.YCrCb2RGB (Y, Cr, Cb)
    frame = cv2.resize(frame, (dimension[1], dimension[0]))
    self.im_save(frame, "Descompressed", f"frame{c + 1}", "jpeg")

  # get decoded frames and make a video from them
  def generate_the_video_from_images(self, c):
    img_array = []
    for i in range(1, c, 2):
      img = cv2.imread(f"Descompressed/frame{i}.jpeg")
      height, width, layers = img.shape
      size = (width, height)
      img_array.append(img)

    if not os.path.exists("Video"):
      os.makedirs("Video")

    out = cv2.VideoWriter("Video/test.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
      out.write(img_array[i])
    out.release()

  # delete files from disk
  def filesRemove(self, path, format):
    py_files = glob.glob(f'{path}/*.{format}')

    for py_file in py_files:
      try:
          os.remove(py_file)
      except OSError as e:
          print(f"Error:{ e.strerror}")
    os.removedirs(path)

  # run the application
  def run(self):
    # video capture
    cap = cv2.VideoCapture(self.video_path)

    # check - if the video is opening
    if (not cap.isOpened()):
      print("Error to open the video file.")
      return

    c = 0

    while (cap.isOpened()):
      # frame reading
      ret, frame = cap.read()

      # check - frame has been read
      if (ret == True):
        if (c%2 == 0):
          dimension = frame.shape

          # saving the original frame as bitmap image file
          self.im_save(frame, "Original-jpeg", f"frame{c+1}", "jpeg")

          # saving the original frame as bitmap image file
          self.im_save(frame, "Original-bmp", f"frame{c+1}", "bmp")

          # frame encoding
          self.encode(frame)

          # frame decoding
          self.decode(dimension, c)

        c += 1

        if cv2.waitKey(25) & 0xFF == ord('q'): 
          break
      else:
          break

    cap.release()
    cv2.destroyAllWindows()

    self.generate_the_video_from_images(c)

    # remove files
    self.filesRemove("Descompressed", "jpeg")
    self.filesRemove("Original-bmp", "bmp")
    self.filesRemove("Original-jpeg", "jpeg")
    self.filesRemove("Txt", "txt")
    self.filesRemove("Video", "mp4")