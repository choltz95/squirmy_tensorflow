import re
import exrex
import numpy as np
import binascii

# alphabet is A, C, T, G
NUM_DIGITS = 180

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int(binascii.hexlify(text.encode(encoding, errors)), 16))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def binary_encode(s):
    bin_arr = [x[-3:] for x in [t for t in map(text_to_bits,s)]]
    bit_arr = []
    for binstr in bin_arr:
        bit_arr.extend(list(binstr))
    bit_arr = [int(i) for i in bit_arr]
    l = len(bit_arr)
    bit_arr.extend([0 for _ in range(NUM_DIGITS - l)])
    return np.array( bit_arr ) 

# generate positive input

def generate_globe():
    return exrex.getone('(A+(TAC|CAT)A)',limit=4)
def generate_eye_spots():
    return exrex.getone('T(CG*T)*AG',limit=5)
def generate_legs():
    return exrex.getone('CG*T',limit=5)
def generate_head():
    return generate_globe() + generate_eye_spots()    
def generate_bod():
    return generate_globe() + generate_legs()
def generate_squirmy():
    return generate_head() + generate_bod()

# generate negative input

def generate_not_squirmy():
    squirmy = '(A+(TAC|CAT)A)T(CG*T)*(A+(TAC|CAT)A)CG*T'
    not_squirmy = '^((?!(' + squirmy + '))(A|C|T|G))*$'
    ns = exrex.getone(not_squirmy,limit=18)
    while ns.strip() == "":
        ns = exrex.getone(not_squirmy,limit=18)
    return ns

trX_pos = np.array([binary_encode(generate_squirmy()) for _ in range(65536)])
trY_pos = np.array([1 for _ in range(65536)])

trX_neg = np.array([binary_encode(generate_not_squirmy()) for _ in range(65536)])
trY_neg = np.array([0 for _ in range(65536)])

trX = np.concatenate((trX_pos,trX_neg),axis=0)
trY = np.concatenate((trY_pos,trY_neg),axis=0)
trY = np.expand_dims(trY, axis=1)

#print(trX)
#print(trY)
np.save("trY",trY)
np.save("trX",trX)

