# Pure obfuscation/crypto helpers extracted from DeDRM_tools ion.py (GPL v3)
import hashlib, hmac, struct, collections
from kfxtables import *

def pkcs7pad(msg, blocklen):
    paddinglen = blocklen - len(msg) % blocklen
    padding = bchr(paddinglen) * paddinglen
    return msg + padding


def pkcs7unpad(msg, blocklen):
    _assert(len(msg) % blocklen == 0)

    paddinglen = msg[-1]

    _assert(paddinglen > 0 and paddinglen <= blocklen, "Incorrect padding - Wrong key")
    _assert(msg[-paddinglen:] == bchr(paddinglen) * paddinglen, "Incorrect padding - Wrong key")

    return msg[:-paddinglen]





# every VoucherEnvelope version has a corresponding "word" and magic number, used in obfuscating the shared secret
# 4-digit versions use their own obfuscation/scramble. It does not seem to depend on the "word" and number
OBFUSCATION_TABLE = {
    "V1":    (0x00, None),
    "V2":    (0x05, b'Antidisestablishmentarianism'),
    "V3":    (0x08, b'Floccinaucinihilipilification'),
    "V4":    (0x07, b'>\x14\x0c\x12\x10-\x13&\x18U\x1d\x05Rlt\x03!\x19\x1b\x13\x04]Y\x19,\t\x1b'),
    "V5":    (0x06, b'~\x18~\x16J\\\x18\x10\x05\x0b\x07\t\x0cZ\r|\x1c\x15\x1d\x11>,\x1b\x0e\x03"4\x1b\x01'),
    "V6":    (0x09, b'3h\x055\x03[^>\x19\x1c\x08\x1b\rtm4\x02Rp\x0c\x16B\n'),
    "V7":    (0x05, b'\x10\x1bJ\x18\nh!\x10"\x03>Z\'\r\x01]W\x06\x1c\x1e?\x0f\x13'),
    "V8":    (0x09, b"K\x0c6\x1d\x1a\x17pO}Rk\x1d'w1^\x1f$\x1c{C\x02Q\x06\x1d`"),
    "V9":    (0x05, b'X.\x0eW\x1c*K\x12\x12\t\n\n\x17Wx\x01\x02Yf\x0f\x18\x1bVXPi\x01'),
    "V10":   (0x07, b'z3\n\x039\x12\x13`\x06=v;\x02MTK\x1e%}L\x1c\x1f\x15\x0c\x11\x02\x0c\n8\x17p'),
    "V11":   (0x05, b'L=\nhVm\x07go\n6\x14\x06\x16L\r\x02\x0b\x0c\x1b\x04#p\t'),
    "V12":   (0x06, b';n\x1d\rl\x13\x1c\x13\x16p\x14\x07U\x0c\x1f\x19w\x16\x16\x1d5T'),
    "V13":   (0x07, b'I\x05\t\x08\x03r)\x01$N\x0fr3n\x0b062D\x0f\x13'),
    "V14":   (0x05, b"\x03\x02\x1c9\x19\x15\x15q\x1057\x08\x16\x0cF\x1b.Fw\x01\x12\x03\x13\x02\x17S'hk6"),
    "V15":   (0x0A, b'&,4B\x1dcI\x0bU\x03I\x07\x04\x1c\t\x05c\x07%ws\x0cj\t\x1a\x08\x0f'),
    "V16":   (0x0A, b'\x06\x18`h;b><\x06PqR\x02Zc\x034\n\x16\x1e\x18\x06#e'),
    "V17":   (0x07, b'y\r\x12\x08fw.[\x02\t\n\x13\x11\x0c\x11b\x1e8L\x10(\x13<Jx6c\x0f'),
    "V18":   (0x07, b'I\x0b\x0e;\x19\x1aIa\x10s\x19g\\\x1b\x11!\x18yf\x0f\t\x1d7[bSp\x03'),
    "V19":   (0x05, b'\n6>)N\x02\x188\x016s\x13\x14\x1b\x16jeN\n\x146\x04\x18\x1c\x0c\x19\x1f,\x02]'),
    "V20":   (0x08, b'_\r\x01\x12]\\\x14*\x17i\x14\r\t!\x1e;~hZ\x12jK\x17\x1e*1'),
    "V21":   (0x07, b'e\x1d\x19|\ty\x1di|N\x13\x0e\x04\x1bj<h\x13\x15k\x12\x08=\x1f\x16~\x13l'),
    "V22":   (0x08, b'?\x17yi$k7Pc\tEo\x0c\x07\x07\t\x1f,*i\x12\x0cI0\x10I\x1a?2\x04'),
    "V23":   (0x08, b'\x16+db\x13\x04\x18\rc%\x14\x17\x0f\x13F\x0c[\t9\x1ay\x01\x1eH'),
    "V24":   (0x06, b'|6\\\x1a\r\x10\nP\x07\x0fu\x1f\t;\rr`uv\\~55\x11]N'),
    "V25":   (0x09, b'\x07\x14w\x1e;^y\x01:\x08\x07\x1fr\tU#j\x16\x12\x1eB\x04\x16=\x06fZ\x07\x02\x06'),
    "V26":   (0x06, b'\x03IL\x1e"K\x1f\x0f\x1fp0\x01`X\x02z0`\x03\x0eN\x07'),
    "V27":   (0x07, b'Xk\x10y\x02\x18\x10\x17\x1d,\x0e\x05e\x10\x15"e\x0fh(\x06s\x1c\x08I\x0c\x1b\x0e'),
    "V28":   (0x0A, b'6P\x1bs\x0f\x06V.\x1cM\x14\x02\n\x1b\x07{P0:\x18zaU\x05'),
    "V9708": (0x05, b'\x1diIm\x08a\x17\x1e!am\x1d\x1aQ.\x16!\x06*\x04\x11\t\x06\x04?'),
    "V1031": (0x08, b'Antidisestablishmentarianism'),
    "V2069": (0x07, b'Floccinaucinihilipilification'),
    "V9041": (0x06, b'>\x14\x0c\x12\x10-\x13&\x18U\x1d\x05Rlt\x03!\x19\x1b\x13\x04]Y\x19,\t\x1b'),
    "V3646": (0x09, b'~\x18~\x16J\\\x18\x10\x05\x0b\x07\t\x0cZ\r|\x1c\x15\x1d\x11>,\x1b\x0e\x03"4\x1b\x01'),
    "V6052": (0x05, b'3h\x055\x03[^>\x19\x1c\x08\x1b\rtm4\x02Rp\x0c\x16B\n'),
    "V9479": (0x09, b'\x10\x1bJ\x18\nh!\x10"\x03>Z\'\r\x01]W\x06\x1c\x1e?\x0f\x13'),
    "V9888": (0x05, b"K\x0c6\x1d\x1a\x17pO}Rk\x1d'w1^\x1f$\x1c{C\x02Q\x06\x1d`"),
    "V4648": (0x07, b'X.\x0eW\x1c*K\x12\x12\t\n\n\x17Wx\x01\x02Yf\x0f\x18\x1bVXPi\x01'),
    "V5683": (0x05, b'z3\n\x039\x12\x13`\x06=v;\x02MTK\x1e%}L\x1c\x1f\x15\x0c\x11\x02\x0c\n8\x17p'),
}


#common str:  "PIDv3AESAES/CBC/PKCS5PaddingHmacSHA256"
class workspace(object):
  def __init__(self,initial_list):
    self.work=initial_list
  def shuffle(self,shuflist):
    ll=len(shuflist)
    rt=[]
    for i in range(ll):
      rt.append(self.work[shuflist[i]])
    self.work=rt
  def sbox(self,table,matrix,skplist=[]): #table is list of 4-byte integers
    offset=0
    nwork=list(self.work)
    wo=0
    toff=0
    while offset<0x6000:
      uv5=table[toff+nwork[wo+0]]
      uv1=table[toff+nwork[wo+1]+0x100]
      uv2=table[toff+nwork[wo+2]+0x200]
      uv3=table[toff+nwork[wo+3]+0x300]
      moff=0
      if 0 in skplist:
        moff+=0x400
      else:
        nib1=matrix[moff+offset+(uv1>>0x1c)|( (uv5>>0x18)&0xf0)]
        moff+=0x100
        nib2=matrix[moff+offset+(uv3>>0x1c)|( (uv2>>0x18)&0xf0)]
        moff+=0x100
        nib3=matrix[moff+offset+((uv1>>0x18)&0xf) |( (uv5>>0x14)&0xf0)]
        moff+=0x100
        nib4=matrix[moff+offset+((uv3>>0x18)&0xf) |( (uv2>>0x14)&0xf0)]
        moff+=0x100
      rnib1=matrix[moff+offset+nib1*0x10+nib2]
      moff+=0x100
      rnib2=matrix[moff+offset+nib3*0x10+nib4]
      moff+=0x100
      nwork[wo+0]=rnib1*0x10+rnib2
      if 1 in skplist:
        moff+=0x400
      else:
        nib1=matrix[moff+offset+((uv1>>0x14)&0xf)|( (uv5>>0x10)&0xf0)]
        moff+=0x100
        nib2=matrix[moff+offset+((uv3>>0x14)&0xf)|( (uv2>>0x10)&0xf0)]
        moff+=0x100
        nib3=matrix[moff+offset+((uv1>>0x10)&0xf) |( (uv5>>0xc)&0xf0)]
        moff+=0x100
        nib4=matrix[moff+offset+((uv3>>0x10)&0xf) |( (uv2>>0xc)&0xf0)]
        moff+=0x100

      rnib1=matrix[moff+offset+nib1*0x10+nib2]
      moff+=0x100
      rnib2=matrix[moff+offset+nib3*0x10+nib4]
      moff+=0x100
      nwork[wo+1]=rnib1*0x10+rnib2
      if 2 in skplist:
        moff+=0x400
      else:
        nib1=matrix[moff+offset+((uv1>>0xc)&0xf)|( (uv5>>0x8)&0xf0)]
        moff+=0x100
        nib2=matrix[moff+offset+((uv3>>0xc)&0xf)|( (uv2>>0x8)&0xf0)]
        moff+=0x100
        nib3=matrix[moff+offset+((uv1>>0x8)&0xf) |( (uv5>>0x4)&0xf0)]
        moff+=0x100
        nib4=matrix[moff+offset+((uv3>>0x8)&0xf) |( (uv2>>0x4)&0xf0)]
        moff+=0x100
      rnib1=matrix[moff+offset+nib1*0x10+nib2]
      moff+=0x100
      rnib2=matrix[moff+offset+nib3*0x10+nib4]
      moff+=0x100
      nwork[wo+2]=rnib1*0x10+rnib2
      if 3 in skplist:
        moff+=0x400
      else:
        nib1=matrix[moff+offset+((uv1>>0x4)&0xf)|( (uv5)&0xf0)]
        moff+=0x100
        nib2=matrix[moff+offset+((uv3>>0x4)&0xf)|( (uv2)&0xf0)]
        moff+=0x100
        nib3=matrix[moff+offset+((uv1)&0xf)|( (uv5<<4)&0xf0) ]
        moff+=0x100
        nib4=matrix[moff+offset+((uv3)&0xf)|( (uv2<<4)&0xf0) ]
        moff+=0x100
      ##############
      rnib1=matrix[moff+offset+nib1*0x10+nib2]
      moff+=0x100
      rnib2=matrix[moff+offset+nib3*0x10+nib4]
      moff+=0x100
      nwork[wo+3]=rnib1*0x10+rnib2
      offset = offset + 0x1800
      wo+=4
      toff+=0x400
    self.work=nwork
  def lookup(self,ltable):
    for a in range(len(self.work)):
      self.work[a]=ltable[a]
  def exlookup(self,ltable):
    lookoffs=0
    for a in range(len(self.work)):
      self.work[a]=ltable[self.work[a]+lookoffs]
      lookoffs+=0x100
  def mask(self, chunk):
    out=[]
    for a in range(len(chunk)):
      self.work[a]=self.work[a]^chunk[a]
      out.append(self.work[a])
    return out

def process_V9708(st):
  #e9c457a7dae6aa24365e7ef219b934b17ed58ee7d5329343fc3aea7860ed51f9a73de14351c9
  ws=workspace([0x11]*16)
  repl=[0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]
  remln=len(st)
  sto=0
  out=[]
  while(remln>0):
    ws.shuffle(repl)
    ws.sbox(d0x6a06ea70,d0x6a0dab50)
    ws.sbox(d0x6a073a70,d0x6a0dab50)
    ws.shuffle(repl)
    ws.exlookup(d0x6a072a70)
    dat=ws.mask(st[sto:sto+16])
    out+=dat
    sto+=16
    remln-=16;
  return bytes(out)

def process_V1031(st):
  #d53efea7fdd0fda3e1e0ebbae87cad0e8f5ef413c471c3ae81f39222a9ec8b8ed582e045918c
  ws=workspace([0x06,0x18,0x60,0x68,0x3b,0x62,0x3e,0x3c,0x06,0x50,0x71,0x52,0x02,0x5a,0x63,0x03])
  repl=[0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]
  remln=len(st)
  sto=0
  out=[]
  while(remln>0):
    ws.shuffle(repl)
    ws.sbox(d0x6a0797c0,d0x6a0dab50,[3])
    ws.sbox(d0x6a07e7c0,d0x6a0dab50,[3])
    ws.shuffle(repl)
    ws.sbox(d0x6a0797c0,d0x6a0dab50,[3])
    ws.sbox(d0x6a07e7c0,d0x6a0dab50,[3])
    ws.exlookup(d0x6a07d7c0)
    dat=ws.mask(st[sto:sto+16])
    out+=dat
    sto+=16
    remln-=16
    #break
  return bytes(out)

def process_V2069(st):
  #8e6196d754a304c9354e91b5d79f07b048026d31c7373a8691e513f2c802c706742731caa858
  ws=workspace([0x79,0x0d,0x12,0x08,0x66,0x77,0x2e,0x5b,0x02,0x09,0x0a,0x13,0x11,0x0c,0x11,0x62])
  repl=[0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]
  remln=len(st)
  sto=0
  out=[]
  while(remln>0):
    ws.sbox(d0x6a084498,d0x6a0dab50,[2])
    ws.shuffle(repl)
    ws.sbox(d0x6a089498,d0x6a0dab50,[2])
    ws.sbox(d0x6a089498,d0x6a0dab50,[2])
    ws.sbox(d0x6a084498,d0x6a0dab50,[2])
    ws.shuffle(repl)
    ws.exlookup(d0x6a088498)
    dat=ws.mask(st[sto:sto+16])
    out+=dat
    sto+=16
    remln-=16
  return bytes(out)


def process_V9041(st):
  #11f7db074b24e560dfa6fae3252b383c3b936e51f6ded570dc936cb1da9f4fc4a97ec686e7d8
  ws=workspace([0x49,0x0b,0x0e,0x3b,0x19,0x1a,0x49,0x61,0x10,0x73,0x19,0x67,0x5c,0x1b,0x11,0x21])
  repl=[0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]
  remln=len(st)
  sto=0
  out=[]
  while(remln>0):
    ws.sbox(d0x6a094170,d0x6a0dab50,[1])
    ws.shuffle(repl)
    ws.shuffle(repl)
    ws.sbox(d0x6a08f170,d0x6a0dab50,[1])
    ws.sbox(d0x6a08f170,d0x6a0dab50,[1])
    ws.sbox(d0x6a094170,d0x6a0dab50,[1])

    ws.exlookup(d0x6a093170)
    dat=ws.mask(st[sto:sto+16])
    out+=dat
    sto+=16
    remln-=16
    #break
  return bytes(out)

def process_V3646(st):
  #d468aa362b44479282291983243b38197c4b4aa24c2c58e62c76ec4b81e08556ca0c54301664
  ws=workspace([0x0a,0x36,0x3e,0x29,0x4e,0x02,0x18,0x38,0x01,0x36,0x73,0x13,0x14,0x1b,0x16,0x6a])
  repl=[0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]
  remln=len(st)
  sto=0
  out=[]
  while(remln>0):
    ws.shuffle(repl)
    ws.sbox(d0x6a099e48,d0x6a0dab50,[2,3])
    ws.sbox(d0x6a09ee48,d0x6a0dab50,[2,3])
    ws.sbox(d0x6a09ee48,d0x6a0dab50,[2,3])
    ws.shuffle(repl)
    ws.sbox(d0x6a099e48,d0x6a0dab50,[2,3])
    ws.sbox(d0x6a099e48,d0x6a0dab50,[2,3])
    ws.shuffle(repl)
    ws.sbox(d0x6a09ee48,d0x6a0dab50,[2,3])
    ws.exlookup(d0x6a09de48)
    dat=ws.mask(st[sto:sto+16])
    out+=dat
    sto+=16
    remln-=16
  return bytes(out)


def process_V6052(st):
  #d683c8c4e4f46ae45812196f37e218eabce0fae08994f25fabb01d3e569b8bf3866b99d36f57
  ws=workspace([0x5f,0x0d,0x01,0x12,0x5d,0x5c,0x14,0x2a,0x17,0x69,0x14,0x0d,0x09,0x21,0x1e,0x3b])
  repl=[0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]
  remln=len(st)
  sto=0
  out=[]
  while(remln>0):
    ws.shuffle(repl)
    ws.sbox(d0x6a0a4b20,d0x6a0dab50,[1,3])
    ws.shuffle(repl)
    ws.sbox(d0x6a0a4b20,d0x6a0dab50,[1,3])
    ws.sbox(d0x6a0a9b20,d0x6a0dab50,[1,3])
    ws.shuffle(repl)
    ws.sbox(d0x6a0a9b20,d0x6a0dab50,[1,3])
    ws.sbox(d0x6a0a9b20,d0x6a0dab50,[1,3])
    ws.sbox(d0x6a0a4b20,d0x6a0dab50,[1,3])

    ws.exlookup(d0x6a0a8b20)
    dat=ws.mask(st[sto:sto+16])
    out+=dat
    sto+=16
    remln-=16
  return bytes(out)

def process_V9479(st):
  #925635db434bccd3f4791eb87b89d2dfc7c93be06e794744eb9de58e6d721e696980680ab551
  ws=workspace([0x65,0x1d,0x19,0x7c,0x09,0x79,0x1d,0x69,0x7c,0x4e,0x13,0x0e,0x04,0x1b,0x6a,0x3c ])
  repl=[0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]
  remln=len(st)
  sto=0
  out=[]
  while(remln>0):
    ws.sbox(d0x6a0af7f8,d0x6a0dab50,[1,2,3])
    ws.sbox(d0x6a0af7f8,d0x6a0dab50,[1,2,3])
    ws.sbox(d0x6a0b47f8,d0x6a0dab50,[1,2,3])
    ws.sbox(d0x6a0af7f8,d0x6a0dab50,[1,2,3])
    ws.shuffle(repl)
    ws.sbox(d0x6a0b47f8,d0x6a0dab50,[1,2,3])
    ws.shuffle(repl)
    ws.shuffle(repl)
    ws.sbox(d0x6a0b47f8,d0x6a0dab50,[1,2,3])
    ws.exlookup(d0x6a0b37f8)

    dat=ws.mask(st[sto:sto+16])
    out+=dat
    sto+=16
    remln-=16
  return bytes(out)

def process_V9888(st):
  #54c470723f8c105ba0186b6319050869de673ce31a5ec15d4439921d4cd05c5e860cb2a41fea
  ws=workspace([0x3f,0x17,0x79,0x69,0x24,0x6b,0x37,0x50,0x63,0x09,0x45,0x6f,0x0c,0x07,0x07,0x09])
  repl=[0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]
  remln=len(st)
  sto=0
  out=[]
  while(remln>0):
    ws.sbox(d0x6a0ba4d0,d0x6a0dab50,[1,2])
    ws.sbox(d0x6a0bf4d0,d0x6a0dab50,[1,2])
    ws.sbox(d0x6a0bf4d0,d0x6a0dab50,[1,2])
    ws.sbox(d0x6a0ba4d0,d0x6a0dab50,[1,2])
    ws.shuffle(repl)
    ws.shuffle(repl)
    ws.shuffle(repl)
    ws.sbox(d0x6a0bf4d0,d0x6a0dab50,[1,2])
    ws.sbox(d0x6a0ba4d0,d0x6a0dab50,[1,2])
    ws.exlookup(d0x6a0be4d0)
    dat=ws.mask(st[sto:sto+16])
    out+=dat
    sto+=16
    remln-=16
  return bytes(out)

def process_V4648(st):
  #705bd4cd8b61d4596ef4ca40774d68e71f1f846c6e94bd23fd26e5c127e0beaa650a50171f1b
  ws=workspace([0x16,0x2b,0x64,0x62,0x13,0x04,0x18,0x0d,0x63,0x25,0x14,0x17,0x0f,0x13,0x46,0x0c])
  repl=[0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]
  remln=len(st)
  sto=0
  out=[]
  while(remln>0):
    ws.sbox(d0x6a0ca1a8,d0x6a0dab50,[1,3])
    ws.shuffle(repl)
    ws.sbox(d0x6a0ca1a8,d0x6a0dab50,[1,3])
    ws.sbox(d0x6a0c51a8,d0x6a0dab50,[1,3])
    ws.sbox(d0x6a0ca1a8,d0x6a0dab50,[1,3])
    ws.sbox(d0x6a0c51a8,d0x6a0dab50,[1,3])
    ws.sbox(d0x6a0c51a8,d0x6a0dab50,[1,3])
    ws.shuffle(repl)
    ws.shuffle(repl)
    ws.exlookup(d0x6a0c91a8)
    dat=ws.mask(st[sto:sto+16])
    out+=dat
    sto+=16
    remln-=16
  return bytes(out)

def process_V5683(st):
  #1f5af733423e5104afb9d5594e682ecf839a776257f33747c9beee671c57ab3f84943f69d8fd
  ws=workspace([0x7c,0x36,0x5c,0x1a,0x0d,0x10,0x0a,0x50,0x07,0x0f,0x75,0x1f,0x09,0x3b,0x0d,0x72])
  repl=[0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]
  remln=len(st)
  sto=0
  out=[]
  while(remln>0):
    ws.sbox(d0x6a0d4e80,d0x6a0dab50,[])
    ws.shuffle(repl)
    ws.sbox(d0x6a0cfe80,d0x6a0dab50,[])
    ws.sbox(d0x6a0d4e80,d0x6a0dab50,[])
    ws.sbox(d0x6a0cfe80,d0x6a0dab50,[])
    ws.sbox(d0x6a0d4e80,d0x6a0dab50,[])
    ws.shuffle(repl)
    ws.sbox(d0x6a0cfe80,d0x6a0dab50,[])
    ws.shuffle(repl)
    ws.exlookup(d0x6a0d3e80)
    dat=ws.mask(st[sto:sto+16])
    out+=dat
    sto+=16
    remln-=16
  return bytes(out)


# def a2hex(arr):
#   ax=[]
#   ha="0123456789abcdef"
#   for a in arr:
#     if a<0: a=256+a
#     ax.append(ha[(a>>4)]+ha[a%16])
#   return "".join(ax)
#
# def memhex(adr,sz):
#   emu=EmulatorHelper(currentProgram)
#   arr=emu.readMemory(getAddress(adr),sz)
#   return a2hex(arr)
#




# obfuscate shared secret according to the VoucherEnvelope version
def obfuscate(secret, version):
    if version == 1:  # v1 does not use obfuscation
        return secret

    magic, word = OBFUSCATION_TABLE["V%d" % version]

    # extend secret so that its length is divisible by the magic number
    if len(secret) % magic != 0:
        secret = secret + b'\x00' * (magic - len(secret) % magic)

    secret = bytearray(secret)

    obfuscated = bytearray(len(secret))
    wordhash = bytearray(hashlib.sha256(word).digest())

    # shuffle secret and xor it with the first half of the word hash
    for i in range(0, len(secret)):
        index = i // (len(secret) // magic) + magic * (i % (len(secret) // magic))
        obfuscated[index] = secret[i] ^ wordhash[index % 16]

    return obfuscated



# scramble() and obfuscate2() from https://github.com/andrewc12/DeDRM_tools/commit/d9233d61f00d4484235863969919059f4d0b2057

def scramble(st,magic):
    ret=bytearray(len(st))
    padlen=len(st)
    for counter in range(len(st)):
        ivar2=(padlen//2)-2*(counter%magic)+magic+counter-1
        ret[ivar2%padlen]=st[counter]
    return ret


def obfuscate2(secret, version):
    if version == 1:  # v1 does not use obfuscation
        return secret
    magic, word = OBFUSCATION_TABLE["V%d" % version]
    # extend secret so that its length is divisible by the magic number
    if len(secret) % magic != 0:
        secret = secret + b'\x00' * (magic - len(secret) % magic)
    obfuscated = bytearray(len(secret))
    wordhash = bytearray(hashlib.sha256(word).digest()[16:])
    #print(wordhash.hex())
    shuffled = bytearray(scramble(secret,magic))
    for i in range(0, len(secret)):
        obfuscated[i] = shuffled[i] ^ wordhash[i % 16]
    return obfuscated

# scramble3() and obfuscate3() from https://github.com/Satsuoni/DeDRM_tools/commit/da6b6a0c911b6d45fe1b13042b690daebc1cc22f

def scramble3(st,magic):
  ret=bytearray(len(st))
  padlen=len(st)
  divs = padlen // magic
  cntr = 0
  iVar6 = 0
  offset = 0
  if (0 < ((magic - 1) + divs)):
    while True:
      if (offset & 1) == 0 :
        uVar4 = divs - 1
        if offset < divs:
          iVar3 = 0
          uVar4 = offset
        else:
          iVar3 = (offset - divs) + 1
        if uVar4>=0:
          iVar5 = uVar4 * magic
          index =  ((padlen - 1) - cntr)
          while True:
            if (magic <= iVar3): break
            ret[index] = st[iVar3 + iVar5]
            iVar3 = iVar3 + 1
            cntr = cntr + 1
            uVar4 = uVar4 - 1
            iVar5 = iVar5 - magic
            index -= 1
            if uVar4<=-1: break
      else:
        if (offset < magic):
          iVar3 = 0
        else :
          iVar3 = (offset - magic) + 1
        if (iVar3 < divs):
          uVar4 = offset
          if (magic <= offset):
            uVar4 = magic - 1

          index = ((padlen - 1) - cntr)
          iVar5 = iVar3 * magic
          while True:
            if (uVar4 < 0) : break
            iVar3 += 1
            ret[index] = st[uVar4 + iVar5]
            uVar4 -= 1
            index=index-1
            iVar5 = iVar5 + magic;
            cntr += 1;
            if iVar3>=divs: break
      offset = offset + 1
      if offset >= ((magic - 1) + divs) :break
  return ret

#not sure if the third variant is used anywhere, but it is in Kindle, so I tried to add it
def obfuscate3(secret, version):
    if version == 1:  # v1 does not use obfuscation
        return secret
    magic, word = OBFUSCATION_TABLE["V%d" % version]
    # extend secret so that its length is divisible by the magic number
    if len(secret) % magic != 0:
        secret = secret + b'\x00' * (magic - len(secret) % magic)
    #secret = bytearray(secret)
    obfuscated = bytearray(len(secret))
    wordhash = bytearray(hashlib.sha256(word).digest())
    #print(wordhash.hex())
    shuffled=bytearray(scramble3(secret,magic))
    #print(shuffled)
    # shuffle secret and xor it with the first half of the word hash
    for i in range(0, len(secret)):
        obfuscated[i] = shuffled[i] ^ wordhash[i % 16]
    return obfuscated
